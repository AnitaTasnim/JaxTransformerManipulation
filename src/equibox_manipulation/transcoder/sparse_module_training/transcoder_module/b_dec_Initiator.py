import equinox as eqx
import jax, jax.numpy as jnp

class b_dec_Initiator:
    
    @staticmethod
    def initialize_b_dec(sparse_module, activation_store):
        if sparse_module.cfg.b_dec_init_method == "geometric_median":
            return b_dec_Initiator.initialize_b_dec_with_geometric_median(sparse_module, activation_store)
        elif sparse_module.cfg.b_dec_init_method == "mean":
            return b_dec_Initiator.initialize_b_dec_with_mean(sparse_module, activation_store)
        elif sparse_module.cfg.b_dec_init_method == "zeros":
            return sparse_module
        else:
            raise ValueError(f"Unexpected b_dec_init_method: {sparse_module.cfg.b_dec_init_method}")

    @staticmethod
    def initialize_b_dec_with_geometric_median(sparse_module, activation_store):
        raise NotImplementedError("geometric median not implemented yet")

    @staticmethod
    def initialize_b_dec_with_mean(sparse_module, activation_store):
        assert sparse_module.cfg.is_transcoder == activation_store.cfg.is_transcoder

        # Current value of b_dec
        previous_b_dec = sparse_module.b_dec
        all_activations = activation_store.storage_buffer
        out = jnp.mean(all_activations, axis=0)

        # Diagnostics
        previous_distances = jnp.linalg.norm(all_activations - previous_b_dec, axis=-1)
        distances = jnp.linalg.norm(all_activations - out, axis=-1)

        # Replace b_dec immutably
        new_module = eqx.tree_at(lambda m: m.b_dec, sparse_module, out.astype(sparse_module.dtype))

        # If transcoder: also update b_dec_out
        if sparse_module.b_dec_out is not None:
            previous_b_dec_out = sparse_module.b_dec_out
            all_activations_out = activation_store.storage_buffer_out
            out_out = jnp.mean(all_activations_out, axis=0)

            previous_distances_out = jnp.linalg.norm(all_activations_out - previous_b_dec_out, axis=-1)
            distances_out = jnp.linalg.norm(all_activations_out - out_out, axis=-1)

            jax.debug.print("Reinitializing b_dec with mean of activations")
            jax.debug.print("Previous distances: {}", jnp.median(previous_distances))
            jax.debug.print("New distances: {}", jnp.median(distances))
            jax.debug.print("Reinitializing b_dec_out with mean of activations")
            jax.debug.print("Previous distances: {}", jnp.median(previous_distances_out))
            jax.debug.print("New distances: {}", jnp.median(distances_out))

            new_module = eqx.tree_at(lambda m: m.b_dec_out, new_module, out_out.astype(sparse_module.dtype))

        return new_module