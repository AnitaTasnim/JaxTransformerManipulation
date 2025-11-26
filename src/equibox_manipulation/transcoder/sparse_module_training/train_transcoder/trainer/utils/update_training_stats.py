import jax.numpy as jnp

def update_training_stats(state):
    """Update counters for tokens, sparsity, and neuron activity."""
    sae_out, feature_acts, mse_loss, l1_loss, ghost_grad_loss, spacon_loss = state["aux"]

    did_fire = jnp.sum((feature_acts > 0), axis=-2) > 0
    state["n_forward_passes_since_fired"] = state["n_forward_passes_since_fired"] + 1
    state["n_forward_passes_since_fired"] = state["n_forward_passes_since_fired"].at[did_fire].set(0)

    state["n_training_tokens"] += state["batch_size"]
    state["act_freq_scores"] += jnp.sum((jnp.abs(feature_acts) > 0), axis=0)
    state["n_frac_active_tokens"] += state["batch_size"]

    return state