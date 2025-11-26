import jax.numpy as jnp
import os
import jax
import equinox as eqx

class BufferManager:

    def __init__(self, cfg, model, tokenizer, dataset_manager):
        self.cfg = cfg
        self.model = model
        self.dataset_manager = dataset_manager
        self.tokenizer = tokenizer 
        self.device = jax.devices(self.cfg.device)[0]

    def _get_activations(self, batch_tokens):
        batch_tokens = jax.device_put(batch_tokens, self.device) #TODO should be on correct device already...

        seq_len = batch_tokens.shape[1]
        position_ids = jnp.arange(seq_len)[None, :]
        position_ids = jnp.broadcast_to(position_ids, batch_tokens.shape)
        position_ids = jax.device_put(position_ids, self.device) #TODO should be on correct device already...

        pad_id = self.tokenizer.pad_token_id or 0
        attention_mask = (batch_tokens != pad_id).astype(jnp.int32)
        attention_mask = jax.device_put(attention_mask, self.device) #TODO should be on correct device already...

        _ = self.model(batch_tokens, position_ids=position_ids, attention_mask=attention_mask)

        act_in = getattr(
            self.model.model.layers[self.cfg.hook_point_layer],
            self.cfg.hook_point,
        ).activation
        act_out = getattr(
            self.model.model.layers[self.cfg.out_hook_point_layer],
            self.cfg.out_hook_point,
        ).activation
        # TODO: ret those via run with cache instead of this hacky way

        return act_in.astype(self.cfg.dtype), act_out.astype(self.cfg.dtype)

    def get_buffer(self, n_batches_in_buffer: int, key: jax.Array):
        """
        Creates a buffer to store activations, either from cache or freshly generated.
        """
        n_batches_in_buffer = self._compute_batches_in_buffer()
        self._assert_no_conflicting_modes()

        if self.cfg.use_cached_activations:
            return self._load_buffer_from_cache(n_batches_in_buffer)
        else:
            return self._generate_buffer_from_tokens(n_batches_in_buffer, key)


    def _compute_batches_in_buffer(self) -> int:
        return self.cfg.n_batches_in_buffer // 2


    def _assert_no_conflicting_modes(self):
        assert not (self.cfg.is_transcoder and self.cfg.use_cached_activations)

    def _load_buffer_from_cache(self, n_batches_in_buffer: int):
        buffer_size = self.cfg.store_batch_size * n_batches_in_buffer * self.cfg.context_size
        new_buffer = jnp.zeros((buffer_size, self.cfg.d_in), dtype=self.cfg.dtype)
        n_tokens_filled = 0

        while n_tokens_filled < buffer_size:
            activations = self._load_next_cache_file(buffer_size, n_tokens_filled)
            if activations is None:  # keine Dateien mehr
                break

            new_buffer, n_tokens_filled = self._insert_into_buffer(new_buffer, activations, n_tokens_filled)

        return new_buffer[:n_tokens_filled]

    def _load_next_cache_file(self, buffer_size, n_tokens_filled):
        path = f"{self.cfg.cached_activations_path}/{self.next_cache_idx}.pt"
        if not os.path.exists(path):
            return None

        import torch
        activations_torch = torch.load(path)
        activations = jnp.array(activations_torch.numpy())

        remaining = buffer_size - n_tokens_filled
        if activations.shape[0] > remaining:
            activations = activations[:remaining]
            self.next_idx_within_buffer = activations.shape[0]
        else:
            self.next_cache_idx += 1
            self.next_idx_within_buffer = 0

        return activations

    def _insert_into_buffer(self, buffer, activations, n_tokens_filled):
        buffer = buffer.at[n_tokens_filled : n_tokens_filled + activations.shape[0]].set(activations)
        return buffer, n_tokens_filled + activations.shape[0]

    def _generate_buffer_from_tokens(self, n_batches_in_buffer: int, key: jax.Array):
        all_tokens = [self.dataset_manager.get_batch_tokens() for _ in range(n_batches_in_buffer)]
        all_tokens = jnp.stack(all_tokens)  # (n_batches, batch_size, context_size)

        if self.cfg.is_transcoder:
            return self._fill_buffer_transcoder(all_tokens, key)
        else:
            return self._fill_buffer_autoencoder(all_tokens, key)

            


    @eqx.filter_jit
    def _fill_buffer_transcoder(self, all_tokens, key):
        """Füllt Buffer im Transcoder-Modus: gibt (in, out) zurück."""
        act_in, act_out = jax.vmap(self._get_activations)(all_tokens)
     
        new_buffer = act_in.reshape(-1, self.cfg.d_in)
        new_buffer_out = act_out.reshape(-1, self.cfg.d_out)

        perm = jax.random.permutation(key, new_buffer.shape[0])
        return new_buffer[perm], new_buffer_out[perm]


    @eqx.filter_jit
    def _fill_buffer_autoencoder(self, all_tokens, key):
        """Füllt Buffer im Autoencoder-Modus: gibt nur in zurück."""
        def body(carry, tokens):
            act_in, _ = self._get_activations(tokens)
            return carry, act_in

        _, all_in = jax.lax.scan(body, None, all_tokens)

        new_buffer = all_in.reshape(-1, self.cfg.d_in)
        perm = jax.random.permutation(key, new_buffer.shape[0])
        return new_buffer[perm]