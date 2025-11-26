from __future__ import annotations
import jax.numpy as jnp
import jax
from equibox_manipulation.transcoder.sparse_module_training.utils import get_device
import wandb
from equibox_manipulation.transcoder.sparse_module_training.train_transcoder.trainer.Losses import Losses

class WandBLogger:

    def log_metrics_to_wandb(state, dead_feature_threshold, wandb_log_frequency):
        """Log losses, variance, sparsity metrics to wandb."""
        sa = state["sparse_autoencoder"]
        sae_out, feature_acts, mse_loss, l1_loss, ghost_grad_loss, spacon_loss = state["aux"]
        sae_in, _ = state["last_batch"]

        # Metrics
        l0 = jnp.mean(jnp.sum((feature_acts > 0), axis=-1))
        current_learning_rate = float(state["schedule"](state["n_training_steps"]) * state["lr_multiplier"])

        per_token_l2_loss = jnp.sum((sae_out - sae_in) ** 2, axis=-1).squeeze()
        total_variance = jnp.sum(sae_in**2, axis=-1)
        explained_variance = 1.0 - per_token_l2_loss / total_variance

        feature_sparsity = state["act_freq_scores"] / state["n_frac_active_tokens"]
        ghost_grad_neuron_mask = (state["n_forward_passes_since_fired"] > sa.cfg.dead_feature_window)

        wandb.log(
            {
                "losses/mse_loss": float(mse_loss),
                "losses/l1_loss": float(l1_loss / sa.l1_coefficient) if sa.l1_coefficient else 0,
                "losses/ghost_grad_loss": float(ghost_grad_loss),
                "losses/overall_loss": float(state["loss_last"]),
                # variance explained
                "metrics/explained_variance": float(jnp.mean(explained_variance)),
                "metrics/explained_variance_std": float(jnp.std(explained_variance)),
                "metrics/l0": float(l0),
                # sparsity
                "sparsity/mean_passes_since_fired": float(jnp.mean(state["n_forward_passes_since_fired"])),
                "sparsity/n_passes_since_fired_over_threshold": int(jnp.sum(ghost_grad_neuron_mask)),
                "sparsity/below_1e-5": float(jnp.mean((feature_sparsity < 1e-5))),
                "sparsity/below_1e-6": float(jnp.mean((feature_sparsity < 1e-6))),
                "sparsity/dead_features": float(jnp.mean((feature_sparsity < dead_feature_threshold))),
                # details
                "details/n_training_tokens": int(state["n_training_tokens"]),
                "details/current_learning_rate": current_learning_rate,
            },
            step=int(state["n_training_steps"]),
        )
        return state


    def log_feature_sparsity_stats(state, use_wandb):
        """Compute and log feature sparsity histogram."""
        feature_sparsity = state["act_freq_scores"] / state["n_frac_active_tokens"]
        log_feature_sparsity = jnp.log10(feature_sparsity + 1e-10)
        log_mean_feature_sparsity = jnp.log10(jnp.mean(feature_sparsity) + 1e-10)

        if use_wandb:
            wandb_histogram = wandb.Histogram(log_feature_sparsity.tolist())
            wandb.log(
                {
                    "metrics/mean_log10_feature_sparsity": float(jnp.mean(log_feature_sparsity)),
                    "metrics/log10_mean_feature_sparsity": float(log_mean_feature_sparsity),
                    "plots/feature_density_line_chart": wandb_histogram,
                },
                step=int(state["n_training_steps"]),
            )

        state["act_freq_scores"] = jnp.zeros((state["sparse_autoencoder"].cfg.d_sae,))
        state["n_frac_active_tokens"] = 0
        state["log_feature_sparsity"] = log_feature_sparsity
        return state

    
    def _log_norm_metrics(
        cfg,
        original_activations,
        sae_reconstruction,
        recons_score,
        base_loss,
        recons_loss,
        zero_ablation_loss,
        n_training_steps,
    ):

        l2_norm_in = jnp.linalg.norm(original_activations, axis=-1)
        l2_norm_out = jnp.linalg.norm(sae_reconstruction, axis=-1)
        l2_norm_out = jax.device_put(l2_norm_out, get_device(cfg.device))
        l2_norm_in = jax.device_put(l2_norm_in, get_device(cfg.device))
        l2_norm_ratio = l2_norm_out / (l2_norm_in + 1e-10)
        

        # Log to wandb
        wandb.log(
            {
                "metrics/l2_norm": float(jnp.mean(l2_norm_out)),
                "metrics/l2_ratio": float(jnp.mean(l2_norm_ratio)),
                "metrics/CE_loss_score": float(recons_score),
                "metrics/ce_loss_without_sparse_module": float(base_loss),
                "metrics/ce_loss_with_sparse_module": float(recons_loss),
                "metrics/ce_loss_with_ablation": float(zero_ablation_loss),
            },
            step=n_training_steps,
        )


    def _log_kl_divergence(sparse_module, original_act, reconstructed_act, ablated_act, n_training_steps):
        head_index = sparse_module.cfg.hook_point_head_index
        if head_index is None:
            return

        kl_result_reconstructed = Losses.kl_divergence_attention(original_act, reconstructed_act) #TODO this function is poorly grouped in Losses
        kl_result_ablation = Losses.kl_divergence_attention(original_act, ablated_act)

        wandb.log(
            {
                "metrics/kldiv_reconstructed": float(jnp.mean(kl_result_reconstructed)),
                "metrics/kldiv_ablation": float(jnp.mean(kl_result_ablation)),
            },
            step=n_training_steps,
        )