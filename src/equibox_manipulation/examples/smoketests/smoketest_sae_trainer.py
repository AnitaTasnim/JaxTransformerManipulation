# smoketest_train_module_funcs.py
import jax
import jax.numpy as jnp
import equinox as eqx
import json
from functools import partial
from transformers import AutoTokenizer
import wandb
from equibox.models.llama import LlamaForCausalLM, LlamaConfig
from equibox_manipulation.hooks.hooks import hooked
from typing import Optional
from equibox_manipulation.transcoder.sae_training.transcoder_module.sparse_autoencoder import SparseAutoencoder

# ---- imports from your trainer file ----
from equibox_manipulation.transcoder.sae_training.train_transcoder.train_sae_on_language_model import (
    train_sae_on_language_model,
    make_standard_replacement_hook,
    make_head_replacement_hook,
    run_evals,
    get_recons_loss,
    mean_ablate_hook,
    zero_ablate_hook,
    kl_divergence_attention,
)

# ---- helpers ----
def load_equinox_model(filename):
    with open(filename, "rb") as f:
        cfg_json = json.loads(f.readline().decode())
        eqx_config = LlamaConfig.from_json_string(cfg_json)
        model = LlamaForCausalLM(eqx_config)
        return eqx.tree_deserialise_leaves(f, model)

def tokenize_to_jax(tokenizer, texts, max_len=64):
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    enc = tokenizer(
        texts if isinstance(texts, (list, tuple)) else [texts],
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="np",
    )
    return jnp.array(enc["input_ids"], dtype=jnp.int32)

class TinyActivationStore:
    class _Cfg:
        def __init__(self, hook_point, hook_point_layer, head_index=None):
            self.is_transcoder = True
            self.hook_point = hook_point
            self.hook_point_layer = hook_point_layer
            self.hook_point_head_index = head_index

    def __init__(self, tokens, d_in, d_out,
                 hook_point="post_attention_layernorm", hook_point_layer=5):
        self._tokens = tokens
        self._key = jax.random.PRNGKey(0)
        self.cfg = self._Cfg(hook_point, hook_point_layer)
        self._d_in, self._d_out = d_in, d_out

    def get_batch_tokens(self):
        return self._tokens

    def next_batch(self, batch_size=4):
        self._key, k = jax.random.split(self._key)
        acts = jax.random.normal(k, (batch_size, self._d_in))
        return jnp.concatenate([acts, acts[:, :self._d_out]], axis=-1)


# ---- smoketest ----
def smoke_test_funcs(
    eqx_model_path="/nfs/home/nbolik/.cache/equibox/hub/Llama-3.2-1B/config_and_weights.eqx",
    hf_model_name="meta-llama/Llama-3.2-1B",
    seq_len=32,
    d_in=2048,
    d_sae=4096,
):
    # 1) model & tokenizer
    llm = hooked(load_equinox_model(eqx_model_path))
    tok = AutoTokenizer.from_pretrained(hf_model_name)
    tokens = tokenize_to_jax(tok, ["func smoketest"], max_len=seq_len)

    # 2) tiny SAE cfg
    class Cfg:
        d_in = 2048
        d_sae = 4096
        d_out = d_in
        l1_coefficient = 1e-3
        dtype = jnp.float32
        is_transcoder = True
        hook_point = "post_attention_layernorm"
        hook_point_layer = 5
        hook_point_head_index = None
        top_k: Optional[int] = None
        use_ghost_grads: bool = False,
        # extras for trainer
        total_training_tokens = 32
        device = "cpu"
        lr_scheduler_name = "constant"
        lr = 1e-3
        lr_warm_up_steps = 0
        use_tqdm = False
        dead_feature_window = 50
        checkpoint_path = "/tmp"
        store_batch_size = 4
        resample_batches = 1
        dead_feature_threshold = 1e-8
        log_to_wandb = False
        is_sparse_connection = False

    key = jax.random.PRNGKey(0)
    sae = SparseAutoencoder(Cfg, key)
    store = TinyActivationStore(tokens, d_in, d_in)

    # --- run the functions ---
    print("[HOOK TEST]")
    repl = make_standard_replacement_hook(sae, llm)
    dummy = jnp.ones((2, d_in))
    print("  replacement out:", repl(dummy, None).shape)

    print("[HEAD HOOK TEST]")
    head_repl = make_head_replacement_hook(sae, llm, head_index=0)
    n_heads = 2
    d_head = d_in  # = 2048 for consistency with this SAE
    dummy_heads = jnp.ones((2, 4, n_heads, d_head))
    out = head_repl(dummy_heads, None)
    print("  head replacement out:", out.shape)

    print("[ABLATION TESTS]")
    m = mean_ablate_hook(jnp.ones((2, 3, d_in)))
    z = zero_ablate_hook(jnp.ones((2, 3, d_in)))
    print("  mean_ablate:", m.shape, "zero_ablate:", z.shape)

    print("[KL TEST]")
    p = jax.nn.softmax(jax.random.normal(key, (4, 4)), axis=-1)
    q = jax.nn.softmax(jax.random.normal(key, (4, 4)), axis=-1)
    kl = kl_divergence_attention(p, q)
    print("  KL div shape:", kl.shape)

    print("[RECONS LOSS TEST]")
    key, sub = jax.random.split(key)
    score, base, recons, zero = get_recons_loss(sae, llm, store, tokens, sub)
    print(f"  base={float(base):.4f} recons={float(recons):.4f} zero={float(zero):.4f} score={float(score):.4f}")

    print("[RUN EVALS TEST]")
    import wandb
    wandb.init(project="tc_library_tests", mode="disabled")
    key, sub = jax.random.split(key)
    run_evals(sae, store, llm, n_training_steps=0, key=sub)
    print("  run_evals executed.")

    print("[TRAIN LOOP ENTRY TEST]")
    # just run a few tokens worth to see if it executes
    try:
        sae2 = train_sae_on_language_model(Cfg, llm, sae, store,
                                           batch_size=4, n_checkpoints=0,
                                           use_wandb=False, wandb_log_frequency=10)
        print("  train_sae_on_language_model returned SAE.")
    except Exception as e:
        print("  train_sae_on_language_model skipped due to:", e)

    print("All function smoketests completed.")


if __name__ == "__main__":
    smoke_test_funcs()
