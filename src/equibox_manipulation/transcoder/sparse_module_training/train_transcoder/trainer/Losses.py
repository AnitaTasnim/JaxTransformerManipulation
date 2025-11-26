import stat
from turtle import st
import jax.numpy as jnp
import jax
import optax
import equinox as eqx
from typing import Tuple
from typing import Any

from equibox_manipulation.transcoder.sparse_module_training.train_transcoder.HookFactory import HookFactory
from equibox_manipulation.transcoder.sparse_module_training.train_transcoder.trainer.utils.Forewarder import Forewarder
from equibox_manipulation.hooks.hooks import hooked
from equibox_manipulation.transcoder.sparse_module_training.make_llama_inputs import make_llama_inputs


class Losses:
    """
    Collection of static loss functions for training and evaluating sparse modules
    (e.g. SparseAutoencoder or SparseTranscoder).
    """

    @staticmethod
    def compute_ce_loss(
        sparse_module: eqx.Module,
        model: eqx.Module,
        batch_tokens: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Compute the cross-entropy loss (next-token prediction) for a given batch.
        """
        input_ids, attention_mask, position_ids = make_llama_inputs(batch_tokens) #TODO generalize to other models
        logits = model(input_ids, attention_mask=attention_mask, position_ids=position_ids)
        labels = input_ids
        return optax.softmax_cross_entropy_with_integer_labels(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1)
        ).mean()

    @staticmethod
    def compute_reconstruction_losses(
        sparse_autoencoder: eqx.Module,
        model: eqx.Module,
        activation_store,
        batch_tokens: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Compute reconstruction-related losses with and without hooks.
        """
        hook_point = activation_store.cfg.hook_point

        base_cross_entropy_loss = Losses.compute_ce_loss(
            sparse_autoencoder, model, batch_tokens
        )

        head_index = sparse_autoencoder.cfg.hook_point_head_index
        if head_index is None:
            replacement_hook = HookFactory.standard_replacement_hook(sparse_autoencoder)
        else:
            replacement_hook = HookFactory.head_replacement_hook(sparse_autoencoder, head_index)

        model_with_replacement = hooked(model, lambda module: replacement_hook if getattr(module, "name", None) == hook_point else None)
        reconstruction_cross_entropy_loss = Losses.compute_ce_loss(sparse_autoencoder, model_with_replacement, batch_tokens)

        model_with_zero_ablation = hooked(model, lambda module: HookFactory.zero_ablate_hook if getattr(module, "name", None) == hook_point else None)
        zero_ablation_cross_entropy_loss = Losses.compute_ce_loss(sparse_autoencoder, model_with_zero_ablation, batch_tokens)


        reconstruction_score = ((zero_ablation_cross_entropy_loss - reconstruction_cross_entropy_loss) / (zero_ablation_cross_entropy_loss - base_cross_entropy_loss + 1e-10))

        return reconstruction_score, base_cross_entropy_loss, reconstruction_cross_entropy_loss, zero_ablation_cross_entropy_loss
        


    @staticmethod
    def kl_divergence_attention(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
        """
        Compute KL divergence between two distributions.
        """
        eps = 1e-10
        log_y_true = jnp.log2(y_true + eps)
        log_y_pred = jnp.log2(y_pred + eps)
        return y_true * (log_y_true - log_y_pred)
    

    @staticmethod
    def compute_loss(sparse_module: eqx.Module, inputs: jnp.ndarray, ghost_grad_mask: jnp.ndarray, target: jnp.ndarray, key: jax.Array) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, ...]]:
        sparse_module_out, feature_acts, total_loss, mse_loss, l1_loss, ghost_grad_loss = Forewarder.forward_train(
            sparse_module, inputs, ghost_grad_mask, target)
        
        sparse_connection_loss = Losses.get_sparse_connection_loss(sparse_module) if sparse_module.cfg.is_sparse_connection else 0.0
        total_loss = total_loss + sparse_connection_loss
        #jax.debug.print("loss = {}, mse_loss ={}, l1_loss = {}", total_loss, mse_loss, l1_loss)
        return total_loss, (sparse_module_out, feature_acts, mse_loss, l1_loss, ghost_grad_loss, sparse_connection_loss)

    @staticmethod
    def get_sparse_connection_loss(sparse_module):
        """
        JAX/eqx port of:
            dots = spacon_sae_W_dec @ W_dec.T
            loss = torch.sum(dots.abs(), dim=1).mean()
            return coeff * loss
        """
        # if not enabled, no-op
        if not getattr(sparse_module.cfg, "is_sparse_connection", False):
            return jnp.array(0.0, dtype=sparse_module.dtype)

        if sparse_module.spacon_sae_W_dec is None:
            raise ValueError(
                "spacon_sae_W_dec is None. Provide reference decoder weights "
                "(e.g., from a pretrained SAE) before calling get_sparse_connection_loss()."
            )

        W_ref = sparse_module.spacon_sae_W_dec.astype(sparse_module.dtype)  # [n_ref_sae, d_out]
        W_cur = sparse_module.W_dec.astype(sparse_module.dtype)             # [d_sae, d_out]

        dots = W_ref @ W_cur.T                            # [n_ref_sae, d_sae]
        loss = jnp.mean(jnp.sum(jnp.abs(dots), axis=1))   # mean over ref features

        return sparse_module.cfg.sparse_connection_l1_coeff * loss 

   
    
    def get_test_loss(self, input_ids, attention_mask, position_ids, llm):
        """
        Cross-entropy pro Token, wenn die MLP des Ziel-Layers durch den Transcoder (SAE) ersetzt wird.
        Signatur passt zu deinem Aufruf: (input_ids, attention_mask, position_ids, llm).
        """
        if not self.cfg.is_transcoder:
            raise NotImplementedError("Non-transcoder case not yet implemented for EQX.")
        assert "mlp" in self.cfg.out_hook_point, "Only MLP transcoders supported"

        # SAE als Wrapper in die MLP des Hook-Layers einsetzen (immutabel)
        old_block = llm.model.layers[int(self.cfg.hook_point_layer)]

        class TranscoderWrapper(eqx.Module):
            transcoder: Any
            def __call__(self, x):
                new_x, *_ = self.transcoder(x)
                return new_x

        new_block = eqx.tree_at(lambda lyr: lyr.mlp, old_block, TranscoderWrapper(self))
        new_llm = eqx.tree_at(lambda m: m.model.layers[int(self.cfg.hook_point_layer)], llm, new_block)

        # Forward + NLL
        logits = new_llm(input_ids, attention_mask, position_ids)          # (B, T, V)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        nll = -jnp.take_along_axis(log_probs, input_ids[..., None], axis=-1).squeeze(-1)  # (B, T)
        return nll