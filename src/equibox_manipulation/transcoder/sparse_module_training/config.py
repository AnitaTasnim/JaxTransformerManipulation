from abc import ABC
from dataclasses import dataclass
from typing import Optional
import jax
import wandb
import jax.numpy as jnp


@dataclass
class RunnerConfig(ABC):
    """
    The config that's shared across all runners (JAX/Equinox version).
    """

    # Data Generating Function (Model + Training Distribution)
    model_name: str = "llama-1b"
    hook_point: str = "layers.0.mlp"         # Equibox-native path
    hook_point_layer: int = 0
    hook_point_head_index: Optional[int] = None
    dataset_path: str = "NeelNanda/c4-tokenized-2b"
    is_dataset_tokenized: Optional[bool] = None
    context_size: int = 128
    use_cached_activations: bool = False
    cached_activations_path: Optional[str] = None
    # Defaults to "activations/{dataset}/{model}/{hook_point}_{hook_point_head_index}"

    # SAE Parameters
    d_in: int = 512

    # Activation Store Parameters
    n_batches_in_buffer: int = 20
    total_training_tokens: int = 2_000_000
    store_batch_size: int = 1024
    data_column: str = "tokens"
    improve_mixing: bool = True

    # Misc
    device: str = "cpu"           # "cpu", "gpu", "tpu" (string only; placement handled by JAX)
    model_device: str = "cpu"
    model_n_devices: int = 1
    seed: int = 42
    dtype = jnp.float32
    model_dtype = jnp.float32
    lazy_device_loading: bool = False

    # Transcoder stuff
    is_transcoder: bool = False
    out_hook_point: Optional[str] = None
    out_hook_point_layer: Optional[int] = None
    d_out: Optional[int] = None

    # Sparse-connection sparse transcoder stuff
    is_sparse_connection: bool = False
    sparse_connection_sae_path: Optional[str] = None
    sparse_connection_l1_coeff: Optional[float] = None
    sparse_connection_use_W_enc: bool = True

    def __post_init__(self):
        # Autofill cached_activations_path unless the user overrode it
        if self.cached_activations_path is None:
            self.cached_activations_path = (
                f"activations/{self.dataset_path.replace('/', '_')}/"
                f"{self.model_name.replace('/', '_')}/"
                f"{self.hook_point}"
            )
            if self.hook_point_head_index is not None:
                self.cached_activations_path += f"_{self.hook_point_head_index}"

@dataclass
class LanguageModelSAERunnerConfig(RunnerConfig):
    """
    Configuration for training a sparse autoencoder on a language model (JAX/Equinox version).
    """

    # SAE Parameters
    b_dec_init_method: str = "geometric_median"   # {"geometric_median", "mean", "zeros"}
    expansion_factor: int = 4
    from_pretrained_path: Optional[str] = None

    # Training Parameters
    l1_coefficient: float = 1e-3
    lr: float = 3e-4
    lr_scheduler_name: str = "constant"           # {"constant", "constantwithwarmup", "linearwarmupdecay", "cosineannealing", "cosineannealingwarmup"}
    lr_warm_up_steps: int = 500
    train_batch_size: int = 4096
    per_device_batch_size: Optional[int] = None

    # Transcoder stuff
    is_transcoder: bool = False
    out_hook_point: Optional[str] = None          # Equibox path, e.g. "layers.5.mlp"
    out_hook_point_layer: Optional[int] = None
    d_out: Optional[int] = None

    # Sparse-connection sparse transcoder stuff
    is_sparse_connection: bool = False
    sparse_connection_sae_path: Optional[str] = None
    sparse_connection_l1_coeff: Optional[float] = None
    sparse_connection_use_W_enc: bool = True
    top_k: Optional[int] = None

    # Resampling protocol args
    use_ghost_grads: bool = False                 # can be toggled later
    feature_sampling_window: int = 2000
    feature_sampling_method: Optional[str] = "anthropic"    # {"none", "l2", "anthropic"}
    resample_batches: int = 32
    feature_reinit_scale: float = 0.2
    dead_feature_window: int = 1000
    dead_feature_estimation_method: str = "no_fire"
    dead_feature_threshold: float = 1e-8

    # WANDB
    log_to_wandb: bool = True
    wandb_project: str = "mats_sae_training_language_model"
    wandb_entity: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_log_frequency: int = 10

    # Misc
    n_checkpoints: int = 0
    checkpoint_path: str = "checkpoints"
    use_tqdm: bool = True

    # Derived attributes (filled in post_init)
    d_sae: Optional[int] = None
    tokens_per_buffer: Optional[int] = None
    run_name: Optional[str] = None

    seed: int = 0

    def __post_init__(self):
        super().__post_init__()
        self.d_sae = self.d_in * self.expansion_factor
        self.tokens_per_buffer = (
            self.train_batch_size * self.context_size * self.n_batches_in_buffer
        )
        self.run_name = f"{self.d_sae}-L1-{self.l1_coefficient}-LR-{self.lr}-Tokens-{self.total_training_tokens:3.3e}"

        # Normalize feature sampling method
        if isinstance(self.feature_sampling_method, str):
            self.feature_sampling_method = self.feature_sampling_method.lower()

        if self.feature_sampling_method not in [None, "l2", "anthropic"]:
            raise ValueError(
                f"feature_sampling_method must be None, l2, or anthropic. Got {self.feature_sampling_method}"
            )

        if self.b_dec_init_method not in ["geometric_median", "mean", "zeros"]:
            raise ValueError(
                f"b_dec_init_method must be geometric_median, mean, or zeros. Got {self.b_dec_init_method}"
            )
        if self.b_dec_init_method == "zeros":
            jax.debug.print(
                "Warning: We are initializing b_dec to zeros. This is probably not what you want."
            )

        if self.device in ["cpu", "gpu", "tpu"]:
            self.jax_device = jax.devices(self.device)[0]
        else:
            self.jax_device = jax.devices()[0]

        unique_id = wandb.util.generate_id()
        self.checkpoint_path = f"{self.checkpoint_path}/{unique_id}"

        jax.debug.print(
            f"Run name: {self.d_sae}-L1-{self.l1_coefficient}-LR-{self.lr}-Tokens-{self.total_training_tokens:3.3e}"
        )
        # Print out some useful info:
        n_tokens_per_buffer = (
            self.store_batch_size * self.context_size * self.n_batches_in_buffer
        )
        jax.debug.print(f"n_tokens_per_buffer (millions): {n_tokens_per_buffer / 10**6}")
        n_contexts_per_buffer = self.store_batch_size * self.n_batches_in_buffer
        jax.debug.print(
            f"Lower bound: n_contexts_per_buffer (millions): {n_contexts_per_buffer / 10**6}"
        )

        total_training_steps = self.total_training_tokens // self.train_batch_size
        jax.debug.print(f"Total training steps: {total_training_steps}")

        total_wandb_updates = total_training_steps // self.wandb_log_frequency
        jax.debug.print(f"Total wandb updates: {total_wandb_updates}")

        # how many times will we sample dead neurons?
        n_dead_feature_samples = total_training_steps // self.dead_feature_window
        n_feature_window_samples = total_training_steps // self.feature_sampling_window
        jax.debug.print(
            f"n_tokens_per_feature_sampling_window (millions): "
            f"{(self.feature_sampling_window * self.context_size * self.train_batch_size) / 10**6}"
        )
        jax.debug.print(
            f"n_tokens_per_dead_feature_window (millions): "
            f"{(self.dead_feature_window * self.context_size * self.train_batch_size) / 10**6}"
        )
        if self.feature_sampling_method is not None:
            jax.debug.print(f"We will reset neurons {n_dead_feature_samples} times.")

        if self.use_ghost_grads:
            jax.debug.print("Using Ghost Grads.")

        jax.debug.print(
            f"We will reset the sparsity calculation {n_feature_window_samples} times."
        )
        jax.debug.print(
            f"Number of tokens when resampling: {self.resample_batches * self.store_batch_size}"
        )
        jax.debug.print(
            f"Number tokens in sparsity calculation window: "
            f"{self.feature_sampling_window * self.train_batch_size:.2e}"
        )

    @dataclass
    class CacheActivationsRunnerConfig(RunnerConfig):
        """
        Configuration for caching activations of an LLM.
        """

        shuffle_every_n_buffers: int = 10
        n_shuffles_with_last_section: int = 10
        n_shuffles_in_entire_dir: int = 10
        n_shuffles_final: int = 100

        def __post_init__(self):
            super().__post_init__()
            if self.use_cached_activations:
                raise ValueError(
                    "use_cached_activations should be False when running cache_activations_runner"
                )