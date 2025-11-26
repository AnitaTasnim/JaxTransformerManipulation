from equibox.models.llama import LlamaForCausalLM, LlamaConfig
from transformers import AutoTokenizer
import logging
import torch
import jax.numpy as jnp
import equinox as eqx
import json
from equibox_manipulation.hooks.hooks import hooked



# ------------------------------------------------------------
# Load EQX LLaMA model
# ------------------------------------------------------------
def load_equinox_model(filename):
    with open(filename, "rb") as f:
        cfg_json = json.loads(f.readline().decode())
        eqx_config = LlamaConfig.from_json_string(cfg_json)
        model = LlamaForCausalLM(eqx_config)
        return eqx.tree_deserialise_leaves(f, model)


# ------------------------------------------------------------
# Tokenizer helper: HuggingFace -> JAX
# ------------------------------------------------------------
def tokenize_to_jax(tokenizer, texts, max_len=64):
    enc = tokenizer(
        texts if isinstance(texts, (list, tuple)) else [texts],
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="np",   # HF supports numpy
    )
    input_ids = jnp.array(enc["input_ids"], dtype=jnp.int32)
    attn_mask = jnp.array(enc["attention_mask"], dtype=jnp.int32)
    seq_len = input_ids.shape[1]
    position_ids = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32), input_ids.shape)
    return input_ids, attn_mask, position_ids


# ------------------------------------------------------------
# Dummy config: only Transcoder case supported
# ------------------------------------------------------------
class DummyCfg:
    is_transcoder = True
    hook_point_layer = 5
    out_hook_point_layer = 5
    hook_point = "post_attention_layernorm"  # entspricht Torch "ln2.hook_normalized"
    out_hook_point = "mlp"                   # entspricht Torch "hook_mlp_out"
    d_in = 4096
    d_out = 4096
    device = "cpu"


# ------------------------------------------------------------
# Activation Store
# ------------------------------------------------------------
class ActivationsStore:
    def __init__(self, cfg, model):
        self.cfg = cfg
        self.model = model

    def get_activations(self, batch_tokens, attention_mask=None, position_ids=None):
        # Forward pass so hooked modules populate their .activation
        _ = self.model(
            batch_tokens,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        if self.cfg.is_transcoder:
            act_in = getattr(
                self.model.model.layers[self.cfg.hook_point_layer],
                self.cfg.hook_point,
            ).activation
            act_out = getattr(
                self.model.model.layers[self.cfg.out_hook_point_layer],
                self.cfg.out_hook_point,
            ).activation
            return jnp.array(act_in), jnp.array(act_out)

        else:
            raise NotImplementedError("Only transcoder case supported.")


# ------------------------------------------------------------
# Main test
# ------------------------------------------------------------
if __name__ == "__main__":
    # 1) Load model and hook
    eqx_model = load_equinox_model(
        "/nfs/data/students/nbolik/models/equinox/Llama-3.2-1B/config_and_weights.eqx"
    )
    eqx_hooked_model = hooked(eqx_model)

    # 2) Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    tokenizer.pad_token = tokenizer.eos_token  # fix for missing pad_token

    # 3) Dummy cfg + store
    cfg = DummyCfg()
    store = ActivationsStore(cfg, eqx_hooked_model)

    # 4) Tokenize text
    input_ids, attn_mask, position_ids = tokenize_to_jax(tokenizer, "Hello Heidelberg!")

    # 5) Get activations
    act_in, act_out = store.get_activations(
        input_ids, attention_mask=attn_mask, position_ids=position_ids
    )

    print("Activation before MLP (shape):", act_in.shape)
    print("Activation after MLP (shape):", act_out.shape)
    print("First few values before MLP:", act_in[0, 0, :5])
    print("First few values after MLP:", act_out[0, 0, :5])