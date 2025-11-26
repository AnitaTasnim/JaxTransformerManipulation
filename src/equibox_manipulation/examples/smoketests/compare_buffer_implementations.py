import jax
import jax.numpy as jnp
import numpy as np
import json, jax, jax.numpy as jnp, equinox as eqx
from transformers import AutoTokenizer
from equibox.models.llama import LlamaForCausalLM, LlamaConfig
from equibox_manipulation.hooks.hooks import hooked
from equibox_manipulation.transcoder.sae_training.activations_store.ActivationsStore import ActivationsStore
from equibox_manipulation.transcoder.sae_training.activations_store.ActivationStoreold import ActivationsStore as ActivationsStoreOld
from typing import Optional
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

class DummyCfg:
    # HF Dataset
    tokenizer = None
    dataset_path = "imdb"
    data_column = "text"
    is_dataset_tokenized = None

    # Cache aus, live generieren
    use_cached_activations = False
    cached_activations_path = "./cache"
    total_training_tokens = 50_000

    # Transcoder only (wir lesen 2 Hookpoints)
    is_transcoder = True
    hook_point_layer = 5
    out_hook_point_layer = 5
    hook_point = "post_attention_layernorm"
    out_hook_point = "mlp"
    improve_mixing = True
    hook_point_head_index: Optional[int] = None

    # Dims (müssen zum Modell passen)
    d_in = 2048
    d_out = 2048
    dtype = jnp.float32
    device = "cpu"

    # Größen
    store_batch_size = 20      # wie viele Sequenzen pro Batch (für get_batch_tokens)
    context_size: int = 128        # Tokens pro Sequenz
    n_batches_in_buffer = 20   # wie viele store-Batches in den Buffer
    train_batch_size = 1024      # minibatch Größe aus dem Buffer
    model_device = "cpu"

    seed = 0

def compare_dataloaders(cfg, tokenizer, model=None, seed=42, n_batches=3):
    key = jax.random.PRNGKey(seed)

    # ---- New implementation ----
    store_new = ActivationsStore(cfg, model=model, tokenizer=tokenizer)
    loader_new = store_new.get_data_loader(key)

    # ---- Old implementation ----
    store_old = ActivationsStoreOld(cfg, model=model, tokenizer=tokenizer)
    loader_old = store_old.get_data_loader(key)

    print(f"\nComparing first {n_batches} batches...")
    for i in range(n_batches):
        try:
            batch_new = next(loader_new)
            batch_old = next(loader_old)
        except StopIteration:
            print("Loader exhausted early")
            break

        arr_new = np.array(batch_new)
        arr_old = np.array(batch_old)

        print(f"\nBatch {i}:")
        print("  Shapes:")
        print("    New:", arr_new.shape)
        print("    Old:", arr_old.shape)

        print("  Statistics:")
        print("    New mean:", arr_new.mean(), "std:", arr_new.std())
        print("    Old mean:", arr_old.mean(), "std:", arr_old.std())

        diff = arr_new - arr_old
        print("  Difference summary:")
        print("    Mean abs diff:", np.abs(diff).mean())
        print("    Max abs diff:", np.abs(diff).max())

        print("  Sample (first row, first 10 values):")
        print("    New:", arr_new[0, :10])
        print("    Old:", arr_old[0, :10])


# ---- Model laden ----
def load_equinox_model(filename):
    with open(filename, "rb") as f:
        cfg_json = json.loads(f.readline().decode())
        eqx_config = LlamaConfig.from_json_string(cfg_json)
        model = LlamaForCausalLM(eqx_config)
        return eqx.tree_deserialise_leaves(f, model)

if __name__ == "__main__":
    cfg = DummyCfg()
    eqx_model = load_equinox_model("/nfs/home/nbolik/.cache/equibox/hub/Llama-3.2-1B/config_and_weights.eqx")
    eqx_hooked_model = hooked(eqx_model)

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    cfg.tokenizer = tokenizer

  
    compare_dataloaders(cfg, tokenizer=tokenizer, model=eqx_hooked_model, seed=123, n_batches=3)