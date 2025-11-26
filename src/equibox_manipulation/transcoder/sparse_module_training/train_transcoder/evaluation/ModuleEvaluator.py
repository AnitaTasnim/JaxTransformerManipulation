import jax
import jax.numpy as jnp
import equinox as eqx
import wandb
from equibox_manipulation.transcoder.sparse_module_training.train_transcoder.trainer.Losses import Losses
from equibox_manipulation.transcoder.sparse_module_training.train_transcoder.HookFactory import HookFactory
from equibox_manipulation.transcoder.sparse_module_training.train_transcoder.logger.WandBLogger import WandBLogger
from equibox_manipulation.transcoder.sparse_module_training.run_with_cache import run_with_cache
from equibox_manipulation.transcoder.sparse_module_training.make_llama_inputs import make_llama_inputs

class ModuleEvaluator:
    """Handles evaluation runs and metrics logging."""

    def __init__(self, sparse_module, model, activation_store, logger=None):
        self.sparse_module = sparse_module
        self.model = model
        self.activation_store = activation_store
        self.logger = logger  # optional, e.g. wandb

    def run_evals(
        self,
        n_training_steps: int,
        key,
        training: bool = False,
    ):
        """Orchestrate full evaluation."""
        cfg = self.sparse_module.cfg
        eval_tokens = self._get_eval_tokens()

        recons_score, base_loss, recons_loss, zero_ablation_loss = self._compute_reconstruction_losses(eval_tokens)


        original_activations = self._get_original_activations(eval_tokens)
        sae_reconstruction = self._forward_reconstruction(original_activations, training)

        WandBLogger._log_norm_metrics(
            cfg,
            original_activations,
            sae_reconstruction,
            recons_score,
            base_loss,
            recons_loss,
            zero_ablation_loss,
            n_training_steps,
        )

        acts_reconstructed = self._get_reconstructed_activations(eval_tokens)
        acts_ablation = self._get_ablated_activations(eval_tokens)
        WandBLogger._log_kl_divergence(self.sparse_module, original_activations, acts_reconstructed, acts_ablation, n_training_steps)

    # === helpers ===

    def _get_eval_tokens(self):
        return self.activation_store.dataset_manager.get_batch_tokens()

    def _compute_reconstruction_losses(self, eval_tokens):
        return Losses.compute_reconstruction_losses(self.sparse_module, self.model, self.activation_store, eval_tokens)
    
    def _get_original_activations(self, eval_tokens):
        cfg = self.sparse_module.cfg
        input_ids, attention_mask, position_ids = make_llama_inputs(eval_tokens) #TODO generalize to other models
        _, acts = run_with_cache(
            llm=self.model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            layer_idx=cfg.hook_point_layer,
            hook_attr=cfg.hook_point,
        )
        return acts if cfg.hook_point_head_index is None else acts[:, :, cfg.hook_point_head_index]

    def _forward_reconstruction(self, activations, training):
        sae_out, *_ = self.sparse_module(activations, training=training)
        return sae_out


    def _get_reconstructed_activations(self, eval_tokens):
        """Run the model with a reconstruction hook and return activations."""
        cfg = self.sparse_module.cfg
        head_index = cfg.hook_point_head_index
        replacement_hook = (
            HookFactory.standard_replacement_hook(self.sparse_module)
            if head_index is None
            else HookFactory.head_replacement_hook(self.sparse_module, head_index)
        )

        model_with_replacement = HookFactory.with_node_hook(
            self.model, cfg.hook_point_layer, cfg.hook_point, replacement_hook
        )
        input_ids, attention_mask, position_ids = make_llama_inputs(eval_tokens)
        _, acts_reconstructed = run_with_cache(
            llm=model_with_replacement,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            layer_idx=cfg.hook_point_layer,
            hook_attr=cfg.hook_point,
        )

        if head_index is not None:
            acts_reconstructed = acts_reconstructed[:, :, head_index]

        return acts_reconstructed


    def _get_ablated_activations(self, eval_tokens):
        """Run the model with a zero-ablation hook and return activations."""
        cfg = self.sparse_module.cfg
        head_index = cfg.hook_point_head_index

        model_with_ablation = HookFactory.with_node_hook(
            self.model, cfg.hook_point_layer, cfg.hook_point, HookFactory.zero_ablate_hook
        )
        input_ids, attention_mask, position_ids = make_llama_inputs(eval_tokens)
        _, acts_ablation = run_with_cache(
            llm=model_with_ablation,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            layer_idx=cfg.hook_point_layer,
            hook_attr=cfg.hook_point,
        )

        if head_index is not None:
            acts_ablation = acts_ablation[:, :, head_index]

        return acts_ablation


    

    