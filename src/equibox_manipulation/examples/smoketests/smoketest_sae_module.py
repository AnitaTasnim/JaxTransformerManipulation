import json
import jax
import jax.numpy as jnp
import equinox as eqx
from transformers import AutoTokenizer

from equibox.models.llama import LlamaForCausalLM, LlamaConfig
from equibox_manipulation.hooks.hooks import hooked
from typing import Any
from pprint import pprint
# --- your JAX/EQX SAE class ---
from equibox_manipulation.transcoder.sae_training.transcoder_module.sparse_autoencoder import SparseAutoencoder  # adjust import!

# -------------------------------
# Load EQX LLaMA (as you do)
# -------------------------------
def load_equinox_model(filename):
    with open(filename, "rb") as f:
        cfg_json = json.loads(f.readline().decode())
        eqx_config = LlamaConfig.from_json_string(cfg_json)
        model = LlamaForCausalLM(eqx_config)
        return eqx.tree_deserialise_leaves(f, model)

# -------------------------------
# Tokenizer helper (HF -> JAX)
# -------------------------------
def tokenize_to_jax(tokenizer, texts, max_len=64):
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    #enc = tokenizer(
    #    texts if isinstance(texts, (list, tuple)) else [texts],
    #    padding=True,
    #    truncation=True,
    #    max_length=max_len,
    #    return_tensors="np",
    #)
    enc = tokenizer(
        texts if isinstance(texts, (list, tuple)) else [texts],
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="np",
    )
    return jnp.array(enc["input_ids"], dtype=jnp.int32)


# -------------------------------
# Minimal cfg for transcoder mode
# -------------------------------
class Cfg:
    # SAE core
    d_in = 2048
    d_sae = 4096           # pick your bottleneck
    d_out = 2048
    l1_coefficient = 5e-3
    dtype = jnp.float32
    top_k = None           # or an int
    use_ghost_grads = False

    # Transcoder flags
    is_transcoder = True
    out_hook_point = "mlp"           # required by your `get_test_loss` assert
    hook_point_layer = 5             # which block to wrap
    # for collecting losses (used by collect_anthropic_resampling_losses)
    hook_point = "post_attention_layernorm"
    hook_point_head_index = None

    # Resampling (if you test it)
    resample_batches = 1
    store_batch_size = 2
    context_size = 64
    feature_reinit_scale = 0.2
    dead_feature_threshold = 1e-8

    # Misc (only for naming/logging normally)
    model_name = "llama-eqx"
    device = "cpu"

# ---------------------------------
# Stub ActivationStore (optional)
# ---------------------------------
class TinyActivationStore:
    """Just enough for collect_anthropic_resampling_losses."""
    def __init__(self, cfg, tokens):
        self.cfg = cfg
        self._tokens = tokens
        # For b_dec init paths (won’t be used here, but satisfy attrs)
        self.storage_buffer = None
        self.storage_buffer_out = None

    def get_batch_tokens(self):
        # always return the same batch (ok for smoke test)
        return self._tokens


def trace_modules(model, prefix=""):
    for name, value in model.__dict__.items():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(value, eqx.Module):
            print(f"[TRACE] {full_name}: {type(value).__name__}")
            trace_modules(value, prefix=full_name)
        elif isinstance(value, list):
            for i, el in enumerate(value):
                if isinstance(el, eqx.Module):
                    trace_modules(el, prefix=f"{full_name}[{i}]")

# -------------------------------
# Smoke test
# -------------------------------
def smoke_test_equibox_transcoder(
    eqx_model_path: str = "/nfs/home/nbolik/.cache/equibox/hub/Llama-3.2-1B/config_and_weights.eqx",
    hf_model_name: str = "meta-llama/Llama-3.2-1B",
    seq_len: int = 64,
    batch_size: int = 2,
    d_in: int = 2048,
):
    # 1) Load model + wrap with hooks
    llm = load_equinox_model(eqx_model_path)
    llm = hooked(llm)
    #trace_modules(llm)
    pprint(llm.model.base_module.layers[5].base_module.post_attention_layernorm)


    # 2) Tokenizer + tokens
    tok = AutoTokenizer.from_pretrained(hf_model_name)
    # Fix für fehlendes pad_token
    if tok.pad_token is None:
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({"pad_token": "[PAD]"})

    texts = [
        "This is a quick transcoder smoke test.",
        "We are testing the SAE integration via hooks."
    ][:batch_size]
    batch_tokens = tokenize_to_jax(tok, texts, max_len=seq_len)
    print("batch_tokens shape:", batch_tokens.shape)  # (B, T)

    # 3) SAE init
    cfg = Cfg()
    cfg.d_in = d_in
    cfg.d_out = d_in  # transcoder I/O dims must match your chosen hook
    key = jax.random.PRNGKey(0)
    sae = SparseAutoencoder(cfg, key)
    print("SAE params:",
          "W_enc", sae.W_enc.shape,
          "W_dec", sae.W_dec.shape,
          "b_dec", sae.b_dec.shape,
          "b_dec_out", None if sae.b_dec_out is None else sae.b_dec_out.shape)

    # 4) Pure forward test (no LLM). In transcoder mode you usually pass a separate target.
    key, k1 = jax.random.split(key)
    x = jax.random.normal(k1, (batch_size, cfg.d_in), dtype=cfg.dtype)
    key, k2 = jax.random.split(key)
    target = jax.random.normal(k2, (batch_size, cfg.d_out), dtype=cfg.dtype)

    out, feats, loss, mse, l1, ghost = sae(x, mse_target=target, training=True)
    print("SAE forward:")
    print("  out:", out.shape, "feats:", feats.shape)
    print("  loss components  mse:", float(mse), "l1:", float(l1), "ghost:", float(ghost))

    # 5) LLM x SAE test: get per-token loss with SAE injected into MLP
    input_ids, attention_mask, position_ids = sae.make_llama_inputs(batch_tokens)
    ce_loss_with_recons = sae.get_test_loss(input_ids, attention_mask, position_ids, llm)
    #ce_loss_with_recons = sae.get_test_loss(batch_tokens, llm)
    print("get_test_loss (with SAE) shape:", ce_loss_with_recons.shape)  # (B, T)
    print("sample mean loss:", float(jnp.mean(ce_loss_with_recons)))

    # 6) (Optional) Anthropic resampling collection with tiny store
    store = TinyActivationStore(cfg, batch_tokens)
    key, k3 = jax.random.split(key)
    incs, acts = sae.collect_anthropic_resampling_losses(llm, store, k3)
    print("collect_anthropic_resampling_losses:")
    print("  global_loss_increases:", incs.shape, "global_input_activations:", acts.shape)

    # 7) (Optional) Resample with fake dead neurons
    dead = jnp.arange(min(4, cfg.d_sae))  # pretend first few are dead
    key, k4 = jax.random.split(key)
    new_sae, _ = sae.resample_neurons_anthropic(dead, llm, None, store, k4)
    print("resample_neurons_anthropic ok. (W_dec changed rows:", dead.shape[0], ")")

    print("Smoke test completed.")

# ---- run it ----
if __name__ == "__main__":
    smoke_test_equibox_transcoder(
        eqx_model_path="/nfs/home/nbolik/.cache/equibox/hub/Llama-3.2-1B/config_and_weights.eqx",
        hf_model_name="meta-llama/Llama-3.2-1B",
        seq_len=64,
        batch_size=2,
        d_in=2048,
    )
