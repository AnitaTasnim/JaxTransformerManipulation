from datasets import load_dataset
from equibox_manipulation.transcoder.sparse_module_training.utils import jax_key_to_seed
from equibox_manipulation.transcoder.sparse_module_training.activations_store.BufferManager import BufferManager
from equibox_manipulation.transcoder.sparse_module_training.activations_store.DataLoaderManager import DataLoaderManager
from equibox_manipulation.transcoder.sparse_module_training.activations_store.DatasetManager import DatasetManager
import jax.numpy as jnp
import os
import jax
class ActivationsStore:
    """
    Class for streaming tokens and generating/storing activations
    while training SAEs (JAX version).
    """
    def __init__(self, cfg, model, tokenizer, create_dataloader: bool = True):
        self.cfg = cfg
        self.device = jax.devices(self.cfg.device)[0]
        self.model = jax.tree_util.tree_map(
            lambda x: jax.device_put(x, self.device) if isinstance(x, jax.Array) else x,
            model
        )  # TODO in JAX: probably HookedModule around an eqx model ???
        self.tokenizer = tokenizer
        self.key = jax.random.PRNGKey(cfg.seed) 

        self.dataset_manager = DatasetManager(cfg, tokenizer, self.key)
        self.buffer_manager = BufferManager(cfg, model, tokenizer, self.dataset_manager)
        self.dataloader_manager = DataLoaderManager(cfg, self.buffer_manager)
        # load HuggingFace dataset
        self.key, subkey = jax.random.split(self.key)
        seed = jax_key_to_seed(subkey)

        # streaming dataset with deterministic shuffle
        self.dataset = (
            load_dataset(cfg.dataset_path, split="train", streaming=True)
            .shuffle(seed=seed)
        )
        self.iterable_dataset = iter(self.dataset)

        
        self.key, subkey1, subkey2 = jax.random.split(self.key, 3)
        buffer, buffer_out = self.buffer_manager.get_buffer(cfg.n_batches_in_buffer, subkey1)
        records = jnp.concatenate([buffer, buffer_out], axis=1)

        self.storage_buffer = buffer
        self.storage_buffer_out = buffer_out

        self.dataloader = self.dataloader_manager.get_data_loader(records, subkey2)
        

    def next_batch(self):
        try:
            batch_tf = next(self.dataloader)          # this is a TF EagerTensor
            batch_np = batch_tf.numpy()               # convert to NumPy
            batch_jax = jax.device_put(batch_np, jax.devices(self.cfg.device)[0])
            return batch_jax
        except StopIteration:
            self.key, subkey = jax.random.split(self.key)
            self.dataloader = self.dataloader_manager.get_data_loader(self.storage_buffer, subkey)
            batch_tf = next(self.dataloader)
            batch_np = batch_tf.numpy()
            batch_jax = jax.device_put(batch_np, jax.devices(self.cfg.device)[0])
            return batch_jax