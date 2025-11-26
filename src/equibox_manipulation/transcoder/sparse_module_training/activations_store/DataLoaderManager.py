import jax.numpy as jnp
import jax
import tensorflow as tf
from equibox_manipulation.transcoder.sparse_module_training.utils import jax_key_to_seed

class DataLoaderManager:
    def __init__(self, cfg, buffer_manager):
        self.cfg = cfg
        self.buffer_manager = buffer_manager

        #TODO we had this off branching with improve mixin etc...
        #TODO think about if we might want to reintroduce it...
        #if create_dataloader:
        #    if self.cfg.improve_mixing:
        #        self.key, subkey = jax.random.split(self.key)
        #        self.storage_buffer_out = None
        #        if self.cfg.is_transcoder:
        #            self.storage_buffer, self.storage_buffer_out = self.get_buffer(self.cfg.n_batches_in_buffer // 2, subkey)
        #        else:
        #            self.storage_buffer = self.get_buffer(self.cfg.n_batches_in_buffer // 2, subkey)

        #    self.key, subkey = jax.random.split(self.key)
        #    self.dataloader = self.get_data_loader(subkey)

    def get_data_loader(self, records, key: jax.Array):
        if not self.cfg.is_transcoder:
            raise NotImplementedError("SAE-only mode not sketched yet")

        key, subkey = jax.random.split(key)

        new_buffer, new_buffer_out = self.buffer_manager.get_buffer(self.cfg.n_batches_in_buffer, key)
        records = jnp.concatenate([new_buffer, new_buffer_out], axis=1)
        np_records = jax.device_get(records)
        with tf.device("/CPU:0"):
            dataset = tf.data.Dataset.from_tensor_slices(np_records)

        seed = jax_key_to_seed(subkey)
        dataset = dataset.shuffle(buffer_size=np_records.shape[0], seed=seed, reshuffle_each_iteration=True)
        dataset = dataset.batch(self.cfg.train_batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        #TODO implement improve mixing logic

        return iter(dataset)
