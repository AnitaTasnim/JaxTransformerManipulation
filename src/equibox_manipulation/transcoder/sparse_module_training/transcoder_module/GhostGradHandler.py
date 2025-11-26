import jax
import jax.numpy as jnp

class GhostGradHandler:
    """Handles optional ghost gradient residual computation for dead neurons."""

    def __init__(self, dtype=jnp.float32):
        self.dtype = dtype

    def with_ghostgradients(self, x, sae_out, hidden_pre, dead_neuron_mask, W_dec, mse_loss):
        """Full ghost gradient loss calculation."""
        residual = x - sae_out
        l2_norm_residual = jnp.linalg.norm(residual, axis=-1)

        mask = dead_neuron_mask.astype(hidden_pre.dtype)
        feature_acts_dead = jnp.exp(hidden_pre) * mask  
        W_dec_masked = W_dec * mask[:, None]       

        ghost_out = feature_acts_dead @ W_dec_masked
        l2_norm_ghost = jnp.linalg.norm(ghost_out, axis=-1)

        norm_scaling_factor = l2_norm_residual / (1e-6 + 2 * l2_norm_ghost)
        ghost_out = ghost_out * jax.lax.stop_gradient(norm_scaling_factor)[:, None]

        mse_loss_ghost_resid = ((ghost_out - jax.lax.stop_gradient(residual)) ** 2) / jnp.sqrt(
            jnp.sum(jax.lax.stop_gradient(residual) ** 2, axis=-1, keepdims=True)
        )
        mse_rescaling_factor = jax.lax.stop_gradient(mse_loss / (mse_loss_ghost_resid + 1e-6))
        mse_loss_ghost_resid = mse_rescaling_factor * mse_loss_ghost_resid
        return jnp.mean(mse_loss_ghost_resid)
        
    def without_ghostgradients(self, _):
        """No-op fallback."""
        
        return jnp.array(0.0, dtype=self.dtype)