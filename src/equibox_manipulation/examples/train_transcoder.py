import argparse
import os
from dataclasses import asdict
from transformers import AutoTokenizer
import jax
import jax.numpy as jnp
import wandb
import equinox as eqx
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# JAX/EQX equivalents
from equibox_manipulation.transcoder.sparse_module_training.config import LanguageModelSAERunnerConfig
from equibox_manipulation.transcoder.sparse_module_training.SeassionLoader import LMSparseAutoencoderSessionloader
from equibox_manipulation.transcoder.sparse_module_training.train_transcoder.train_sae_on_language_model import train_sae_on_language_model
from equibox_manipulation.transcoder.sparse_module_training.transcoder_module.Persistor import Persistor



def main(args):
    # ------- Hyperparameters  -------
    lr = 4e-4                     # learning rate
    l1_coeff = 1e-4               # L1 sparsity coefficient
    expansion_factor = 8
    batch_size = 4096
    per_device_batch_size = None
    total_training_tokens = 81_380_000
    l1_warm_up_steps = 5000

    # ------- Build config (must match your JAX dataclass fields) -------
    cfg = LanguageModelSAERunnerConfig(
        # Hook points — Equinox names (layer 5 transcoder)
        hook_point="post_attention_layernorm",
        hook_point_layer=5,
        d_in=2048,

        # Data/model identifiers (these likely live in RunnerConfig base)
        dataset_path="codeparrot/github-code",
        is_dataset_tokenized=False,
        model_name="meta-llama/Llama-3.2-1B",

        # Transcoder specifics
        is_transcoder=True,
        out_hook_point="mlp",
        out_hook_point_layer=5,
        d_out=2048,

        # SAE params
        expansion_factor=expansion_factor,
        b_dec_init_method="mean",

        # Training params
        lr=lr,
        l1_coefficient=l1_coeff,
        lr_scheduler_name="constantwithwarmup",
        train_batch_size=batch_size,
        per_device_batch_size=per_device_batch_size,
        context_size=128,
        lr_warm_up_steps=l1_warm_up_steps,

        # Activation store params
        n_batches_in_buffer=64,#32,    # ensure: n_batches_in_buffer * store_batch_size * context_size >= 2 * train_batch_size
        total_training_tokens=total_training_tokens,
        store_batch_size=4,
        data_column="code",
        improve_mixing=True,

        # Dead neurons / sparsity
        use_ghost_grads=True,
        feature_sampling_method=None,   # "anthropic" if you wired resampling fully
        feature_sampling_window=1000,
        resample_batches=1028,
        dead_feature_window=5000,
        dead_feature_threshold=1e-8,
        top_k=None,

        # WandB
        log_to_wandb=not args.no_wandb,
        wandb_project="Transcoder_Llama1B_JAX",
        wandb_entity="pvs-shared",
        wandb_group=None,
        wandb_log_frequency=10,

        # Misc (NO dtype/model_device fields here)
        use_tqdm=True,
        device="gpu",              # set "gpu" or "cpu" - ther are still some issues of params being initialized on cpu and not device
        seed=42,
        n_checkpoints=0,
        checkpoint_path="/nfs/data/shared/eqx-transcoders/llama-1b/",
    )


    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    # Ensure pad_token is set consistently
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    cfg.tokenizer = tokenizer

    print(f"About to start training with lr {lr} and l1 {l1_coeff}")
    print(f"Checkpoint path: {cfg.checkpoint_path}")
    print(cfg)

    if cfg.log_to_wandb:
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            group=cfg.wandb_group,
            config=asdict(cfg),
            mode="offline" if args.offline else "online",
        )
        run_folder = Path(cfg.checkpoint_path) / wandb.run.id
        os.makedirs(run_folder, exist_ok=True)
    else:
        os.makedirs(Path(cfg.checkpoint_path), exist_ok=True)

    # ------- Session loader (JAX/EQX) -------
    loader = LMSparseAutoencoderSessionloader(cfg)
    model, sparse_autoencoder, activations_loader = loader.load_session()

    # ------- Train SAE -------
    sparse_autoencoder = train_sae_on_language_model(
        cfg,
        model,
        sparse_autoencoder,
        activations_loader,
        n_checkpoints=cfg.n_checkpoints,
        batch_size=cfg.train_batch_size, 
        feature_sampling_method=cfg.feature_sampling_method,
        feature_sampling_window=cfg.feature_sampling_window,
        feature_reinit_scale=cfg.feature_reinit_scale,
        dead_feature_threshold=cfg.dead_feature_threshold,
        dead_feature_window=cfg.dead_feature_window,
        use_wandb=cfg.log_to_wandb,
        wandb_log_frequency=cfg.wandb_log_frequency,
    )

    # ------- Save final SAE (JAX/EQX) -------
    Persistor.save_sae_model(cfg, sparse_autoencoder)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", default=False)
    parser.add_argument("--no_wandb", action="store_true", default=False)
    # Keep these minimal so you can “just run” without args
    args = parser.parse_args()
    main(args)