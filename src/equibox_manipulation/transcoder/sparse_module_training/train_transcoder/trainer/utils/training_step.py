import jax
import jax.numpy as jnp
from equibox_manipulation.transcoder.sparse_module_training.train_transcoder.trainer.ModuleOptimizer import ModuleOptimizer

def training_step(state, activation_store):
    """Perform one training step with gradient accumulation."""
    sa = state["sparse_autoencoder"]
    next_batch = activation_store.next_batch()
    ghost_grad_neuron_mask = (state["n_forward_passes_since_fired"] > sa.cfg.dead_feature_window)

    if not sa.cfg.is_transcoder:
        sae_in, mse_target = next_batch, next_batch
    else:
        sae_in = next_batch[:, :sa.cfg.d_in]
        mse_target = next_batch[:, sa.cfg.d_in:]

    keys_accum = jax.random.split(state["key"], state["gradient_accumulation_steps"])
    state["key"] = jax.random.split(state["key"], 1)[0]

    sa, opt_state, aux, loss_last, loss_mean = ModuleOptimizer.train_step(
        sa, state["opt_state"], sae_in, mse_target,
        ghost_grad_neuron_mask, state["lr_multiplier"], keys_accum, state["optimizer"]
    )

    state["sparse_autoencoder"] = sa
    state["opt_state"] = opt_state
    state["aux"] = aux
    state["loss_last"] = loss_last
    state["loss_mean"] = loss_mean
    state["last_batch"] = (sae_in, mse_target)
    return state

