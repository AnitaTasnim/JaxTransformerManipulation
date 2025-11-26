import json, jax, jax.numpy as jnp, equinox as eqx
from transformers import AutoTokenizer
from equibox.models.llama import LlamaForCausalLM, LlamaConfig
from equibox_manipulation.hooks.hooks import hooked
from equibox_manipulation.transcoder.sae_training.activations_store.ActivationsStore import ActivationsStore
from typing import Optional

# ---- Model laden ----
def load_equinox_model(filename):
    with open(filename, "rb") as f:
        cfg_json = json.loads(f.readline().decode())
        eqx_config = LlamaConfig.from_json_string(cfg_json)
        model = LlamaForCausalLM(eqx_config)
        return eqx.tree_deserialise_leaves(f, model)

# ---- Config ----
class DummyCfg:
    # HF Dataset
    dataset_path = "imdb"
    data_column = "text"
    is_dataset_tokenized = None

    # Cache aus, live generierenmodel
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

if __name__ == "__main__":
    # 1) LLaMA laden und hooken
    eqx_model = load_equinox_model("/nfs/home/nbolik/.cache/equibox/hub/Llama-3.2-1B/config_and_weights.eqx")
    eqx_hooked_model = hooked(eqx_model)

    # 2) Tokenizer (HF)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    # Sicherheitsnetz für BOS/PAD
    # Sicherheitsnetz für BOS/PAD
    if tokenizer.bos_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.bos_token_id = tokenizer.eos_token_id

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    # 3) Store
    cfg = DummyCfg()
    store = ActivationsStore(cfg, eqx_hooked_model, tokenizer, create_dataloader=True)

    # 4) Tokens ziehen
    batch_tokens = store.get_batch_tokens()
    print("Batch tokens:", batch_tokens.shape)
    assert batch_tokens.shape == (cfg.store_batch_size, cfg.context_size)

    act_in, act_out = store._get_activations(batch_tokens)
    print("Activations in:", act_in.shape)
    print("Activations out:", act_out.shape)

    key = jax.random.PRNGKey(0)
    buffer_in, buffer_out = store.get_buffer(2, key)
    print("Buffer in:", buffer_in.shape)
    print("Buffer out:", buffer_out.shape)

    key = jax.random.PRNGKey(42)
    dataloader = store.get_data_loader(key)
    first_batch = next(dataloader)
    print("First minibatch:", first_batch.shape)

    key = jax.random.PRNGKey(42)
    dataloader = store.get_data_loader(key)
    first_batch = next(dataloader)
    print("First minibatch:", first_batch.shape)

    for i, batch in enumerate(store.get_data_loader(key)):
        print(f"Batch {i}:", batch.shape)
        if i == 2: break
