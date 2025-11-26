from typing import Optional
import jax
from equibox_manipulation.hooks.hooks import HookedModule
from equibox_manipulation.transcoder.sparse_module_training.activations_store.ActivationsStore import ActivationsStore
from equibox_manipulation.transcoder.sparse_module_training.transcoder_module.sparse_autoencoder import SparseAutoencoder
from equibox_manipulation.transcoder.sparse_module_training.train_transcoder.trainer.initialize_training_state import initialize_training_state
from equibox_manipulation.transcoder.sparse_module_training.train_transcoder.resamling_methods.FeatureResampler import FeatureResampler
from equibox_manipulation.transcoder.sparse_module_training.train_transcoder.trainer.utils.LearningRateScheduler import LearningRateScheduler
from equibox_manipulation.transcoder.sparse_module_training.train_transcoder.trainer.utils.training_step import training_step
from equibox_manipulation.transcoder.sparse_module_training.train_transcoder.trainer.utils.update_training_stats import update_training_stats
from equibox_manipulation.transcoder.sparse_module_training.train_transcoder.logger.WandBLogger import WandBLogger
from equibox_manipulation.transcoder.sparse_module_training.train_transcoder.logger.save_at_checkpoint import save_at_checkpoint
from equibox_manipulation.transcoder.sparse_module_training.train_transcoder.logger.update_progress_bar import update_progress_bar
from equibox_manipulation.transcoder.sparse_module_training.train_transcoder.evaluation.run_evaluations import run_evaluations
from equibox_manipulation.transcoder.sparse_module_training.train_transcoder.trainer.utils.ModuleNormalizer import ModuleNormalizer

def train_sae_on_language_model(
    cfg, model: HookedModule, sparse_autoencoder: SparseAutoencoder,
    activation_store: ActivationsStore, batch_size: int = 1024,
    per_device_batch_size: Optional[int] = None, n_checkpoints: int = 0,
    feature_sampling_method: str = "l2", feature_sampling_window: int = 1000,
    feature_reinit_scale: float = 0.2, dead_feature_threshold: float = 1e-8,
    dead_feature_window: int = 2000, use_wandb: bool = False,
    wandb_log_frequency: int = 50,
):

    state = initialize_training_state(
        cfg, sparse_autoencoder, activation_store, batch_size, per_device_batch_size,
        feature_sampling_method, feature_sampling_window, dead_feature_window, n_checkpoints
    )

    while state["n_training_tokens"] < state["total_training_tokens"]:
        sa = state["sparse_autoencoder"]

        # Normalization
        sa = ModuleNormalizer.set_decoder_norm_to_unit_norm(sa) #TODO function has to be shifted to a higher level as being called here
        state["sparse_autoencoder"] = sa

        # Resampling
        if state["feature_sampling_method"] == "anthropic" and ((state["n_training_steps"] + 1) % state["dead_feature_window"] == 0):
            state = FeatureResampler.resample_dead_neurons(state, model, activation_store, use_wandb)
            #TODO!!! there is a different definition in module definition...why? I just noticed that

        if state["feature_sampling_method"] == "l2" and ((state["n_training_steps"] + 1) % state["dead_feature_window"] == 0):
            jax.debug.print("no l2 resampling currently. Please use anthropic resampling")
            #TODO!!! there is a definition in module definition...why was it kicked out in torch implement?

        # Feature sparsity logging
        if (state["n_training_steps"] + 1) % state["feature_sampling_window"] == 0:
            state = WandBLogger.log_feature_sparsity_stats(state, use_wandb)

        state = LearningRateScheduler.update_learning_rate(state)
        state = training_step(state, activation_store)
        state = update_training_stats(state)

        if use_wandb and ((state["n_training_steps"] + 1) % wandb_log_frequency == 0):
            state = WandBLogger.log_metrics_to_wandb(state, dead_feature_threshold, wandb_log_frequency)

        state = run_evaluations(state, model, activation_store, wandb_log_frequency)
        state = update_progress_bar(state)
        state = save_at_checkpoint(state, dead_feature_threshold, use_wandb)

        state["n_training_steps"] += 1

    return state["sparse_autoencoder"]