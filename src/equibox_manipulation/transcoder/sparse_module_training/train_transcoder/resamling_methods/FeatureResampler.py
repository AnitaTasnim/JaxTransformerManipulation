import jax
import jax.numpy as jnp
import wandb

class FeatureResampler:
    """Handles dead neuron detection and resampling logic."""
    def __init__(self, sae, cfg):
        self.sae = sae
        self.cfg = cfg
    
    def resample_dead_neurons(self, state, model, activation_store, use_wandb):
        """Detect and resample dead neurons (anthropic method)."""
        sa = state["sparse_autoencoder"]
        key = state["key"]

        feature_sparsity = state["act_freq_scores"] / state["n_frac_active_tokens"]

        if sa.cfg.dead_feature_estimation_method == "no_fire":
            dead_neuron_indices = jnp.where(state["act_freq_scores"] == 0)[0]
        elif sa.cfg.dead_feature_estimation_method == "frequency":
            dead_neuron_indices = jnp.where(feature_sparsity < sa.cfg.dead_feature_threshold)[0]
        else:
            dead_neuron_indices = jnp.array([])

        if dead_neuron_indices.shape[0] > 0:
            key, subkey = jax.random.split(key)
            sa, opt_state = sa.resample_neurons_anthropic(
                dead_neuron_indices, model, activation_store, subkey
            )

            if use_wandb:
                n_resampled = int(
                    jnp.minimum(
                        dead_neuron_indices.shape[0],
                        sa.cfg.store_batch_size * sa.cfg.resample_batches,
                    )
                )
                wandb.log({"metrics/n_resampled_neurons": n_resampled}, step=int(state["n_training_steps"]))
                state["n_resampled_neurons"] = n_resampled

            # Learning rate reset logic
            current_lr = state["schedule"](state["n_training_steps"])
            reduced_lr = current_lr / 10_000.0
            increment = (current_lr - reduced_lr) / 10_000.0
            state["lr_multiplier"] = reduced_lr / current_lr
            state["steps_before_reset"] = 10_000

            state["sparse_autoencoder"] = sa
            state["opt_state"] = opt_state
            state["key"] = key
        return state