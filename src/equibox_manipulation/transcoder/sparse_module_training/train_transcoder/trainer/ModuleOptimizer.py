from equibox_manipulation.transcoder.sparse_module_training.train_transcoder.trainer.utils.ModuleNormalizer import ModuleNormalizer
import jax
import jax.numpy as jnp
import equinox as eqx
from equibox_manipulation.transcoder.sparse_module_training.train_transcoder.trainer.Losses import Losses

class ModuleOptimizer:
    """Encapsulates one optimization/training step for SAE."""


    def micro_step(sae, sae_in, ghost_grad_mask, target, key):
        (loss, aux), grads = jax.value_and_grad(Losses.compute_loss, has_aux=True)(
            sae, sae_in, ghost_grad_mask, target, key
        )
        return loss, aux, grads


    @eqx.filter_jit
    def train_step(
        sae,
        opt_state,
        sae_in,
        mse_target,
        ghost_grad_mask,
        lr_multiplier,      # scalar float
        keys_accum,         # [grad_steps, ...] PRNG keys
        optimizer
    ):
        def _one_grad(key):
            (loss, aux), grads = jax.value_and_grad(Losses.compute_loss, has_aux=True)(
                sae, sae_in, ghost_grad_mask, mse_target, key
            )
            return (loss, aux), grads

        (losses, aux_stacked), grads_stacked = jax.vmap(_one_grad)(keys_accum)

        # average grads for the update (same as your mean_grads)
        mean_grads = jax.tree_util.tree_map(lambda g: g.mean(axis=0), grads_stacked)
        mean_grads = ModuleNormalizer.remove_gradient_parallel_to_decoder_directions(sae, mean_grads)

        updates, opt_state = optimizer.update(mean_grads, opt_state, sae)
        updates = jax.tree_util.tree_map(lambda u: u * lr_multiplier, updates)
        sae = eqx.apply_updates(sae, updates)

        # preserve previous semantics: use last micro-step for logs
        aux_last  = jax.tree_util.tree_map(lambda x: x[-1], aux_stacked)
        loss_last = losses[-1]
        loss_mean = losses.mean()

        return sae, opt_state, aux_last, loss_last, loss_mean
