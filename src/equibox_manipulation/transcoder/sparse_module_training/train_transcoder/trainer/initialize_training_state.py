from equibox_manipulation.transcoder.sparse_module_training.transcoder_module import b_dec_Initiator
import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm
from equibox_manipulation.transcoder.sparse_module_training.utils import get_device
from equibox_manipulation.transcoder.sparse_module_training.train_transcoder.trainer.utils.LearningRateScheduler import LearningRateScheduler
from equibox_manipulation.transcoder.sparse_module_training.transcoder_module.b_dec_Initiator import b_dec_Initiator

def initialize_training_state(cfg, sparse_autoencoder, activation_store, batch_size, per_device_batch_size,
                              feature_sampling_method, feature_sampling_window, dead_feature_window, n_checkpoints):
    """Set up RNG, scheduler, optimizer, counters, and checkpoint thresholds."""
    key = jax.random.PRNGKey(cfg.seed)
    gradient_accumulation_steps = 1

    if per_device_batch_size is not None:
        gradient_accumulation_steps = batch_size // per_device_batch_size
        batch_size = per_device_batch_size
        feature_sampling_window *= gradient_accumulation_steps
        dead_feature_window *= gradient_accumulation_steps

    if feature_sampling_method is not None:
        feature_sampling_method = feature_sampling_method.lower()

    total_training_tokens = sparse_autoencoder.cfg.total_training_tokens
    total_training_steps = total_training_tokens // batch_size

    checkpoint_thresholds = []
    if n_checkpoints > 0:
        checkpoint_thresholds = list(
            range(0, total_training_tokens, total_training_tokens // (n_checkpoints + 1))
        )[1:]

    device = get_device(sparse_autoencoder.cfg.device)
    act_freq_scores = jax.device_put(jnp.zeros((sparse_autoencoder.cfg.d_sae,)), device)
    n_forward_passes_since_fired = jax.device_put(jnp.zeros((sparse_autoencoder.cfg.d_sae,)), device)

    schedule = LearningRateScheduler.get_scheduler(
        sparse_autoencoder.cfg.lr_scheduler_name,
        base_lr=sparse_autoencoder.cfg.lr,
        warm_up_steps=sparse_autoencoder.cfg.lr_warm_up_steps * gradient_accumulation_steps,
        training_steps=total_training_steps,
        lr_end=sparse_autoencoder.cfg.lr / 10,  # heuristic
    )
    optimizer = optax.adam(schedule)
    sparse_autoencoder = b_dec_Initiator.initialize_b_dec(sparse_autoencoder, activation_store)
    opt_state = optimizer.init(sparse_autoencoder)

    pbar = None
    if sparse_autoencoder.cfg.use_tqdm:
        pbar = tqdm(total=total_training_tokens, desc="Training SAE")

    return dict(
        key=key,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        feature_sampling_method=feature_sampling_method,
        feature_sampling_window=feature_sampling_window,
        dead_feature_window=dead_feature_window,
        checkpoint_thresholds=checkpoint_thresholds,
        total_training_tokens=total_training_tokens,
        total_training_steps=total_training_steps,
        n_training_tokens=0,
        n_training_steps=0,
        n_resampled_neurons=0,
        steps_before_reset=0,
        act_freq_scores=act_freq_scores,
        n_forward_passes_since_fired=n_forward_passes_since_fired,
        n_frac_active_tokens=0,
        schedule=schedule,
        optimizer=optimizer,
        opt_state=opt_state,
        sparse_autoencoder=sparse_autoencoder,
        pbar=pbar,
        log_feature_sparsity=None,
    )