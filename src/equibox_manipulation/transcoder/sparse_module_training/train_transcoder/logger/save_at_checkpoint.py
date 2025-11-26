import jax.numpy as jnp
import equinox as eqx
import wandb

def save_at_checkpoint(state, dead_feature_threshold, use_wandb):
    """Save model + sparsity arrays at checkpoints."""
    if len(state["checkpoint_thresholds"]) == 0:
        return state

    if state["n_training_tokens"] > state["checkpoint_thresholds"][0]:
        sa = state["sparse_autoencoder"]
        cfg = sa.cfg

        path = f"{cfg.checkpoint_path}/{state['n_training_tokens']}_{sa.get_name()}.eqx"
        log_feature_sparsity_path = (
            f"{cfg.checkpoint_path}/{state['n_training_tokens']}_{sa.get_name()}_log_feature_sparsity.npy"
        )

        # Save model + log sparsity
        eqx.tree_serialise_leaves(path, sa)
        feature_sparsity = state["act_freq_scores"] / state["n_frac_active_tokens"]
        log_feature_sparsity = jnp.log10(feature_sparsity + 1e-10)
        jnp.save(log_feature_sparsity_path, log_feature_sparsity)

        state["checkpoint_thresholds"].pop(0)
        if len(state["checkpoint_thresholds"]) == 0:
            state["n_checkpoints"] = 0

        if cfg.log_to_wandb and use_wandb:
            model_artifact = wandb.Artifact(
                f"{sa.get_name()}",
                type="model",
                metadata=dict(cfg.__dict__),
            )
            model_artifact.add_file(path)
            wandb.log_artifact(model_artifact)

            sparsity_artifact = wandb.Artifact(
                f"{sa.get_name()}_log_feature_sparsity",
                type="log_feature_sparsity",
                metadata=dict(cfg.__dict__),
            )
            sparsity_artifact.add_file(log_feature_sparsity_path)
            wandb.log_artifact(sparsity_artifact)

    return state