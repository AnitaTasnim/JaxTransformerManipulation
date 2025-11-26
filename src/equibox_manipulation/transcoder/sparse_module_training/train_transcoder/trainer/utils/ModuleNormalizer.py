import jax.numpy as jnp
import equinox as eqx
import einops

class ModuleNormalizer:

    def set_decoder_norm_to_unit_norm(sparse_module):
        """Return new sparse_module with row-normalized decoder weights."""
        W_dec = sparse_module.W_dec / jnp.linalg.norm(sparse_module.W_dec, axis=1, keepdims=True)
        return eqx.tree_at(lambda m: m.W_dec, sparse_module, W_dec)
    
    @staticmethod
    def remove_gradient_parallel_to_decoder_directions(sparse_module, grads):
        """
        Project gradient of W_dec to be orthogonal to W_dec rows.
        Functional version for JAX/eqx.
        """
        W_dec = sparse_module.W_dec
        grad_W_dec = grads.W_dec

        # Compute parallel component per row
        parallel_component = einops.einsum(
            grad_W_dec, W_dec, "d_sparse_module d_out, d_sparse_module d_out -> d_sparse_module"
        )

        correction = einops.einsum(
            parallel_component, W_dec, "d_sparse_module, d_sparse_module d_out -> d_sparse_module d_out"
        )

        grad_W_dec = grad_W_dec - correction

        # Return new grads pytree with corrected W_dec gradient
        new_grads = eqx.tree_at(lambda g: g.W_dec, grads, grad_W_dec)
        return new_grads