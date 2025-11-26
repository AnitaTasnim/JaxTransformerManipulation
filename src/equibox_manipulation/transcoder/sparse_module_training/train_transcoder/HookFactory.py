import jax.numpy as jnp
import equinox as eqx
from typing import Callable, Optional


class HookFactory:
    """
    Factory for creating hook functions used to replace or modify activations
    during model forward passes. 
    """

    # === Replacement Hooks ===

    @staticmethod
    def standard_replacement_hook(sparse_module: eqx.Module) -> Callable[[jnp.ndarray, Optional[object]], jnp.ndarray]:
        """
        Create a hook that replaces activations with SAE reconstructions.
        """
        def hook(activations: jnp.ndarray) -> jnp.ndarray:
            sae_out, *_ = sparse_module(activations)
            return sae_out.astype(activations.dtype)

        return hook

    @staticmethod
    def head_replacement_hook(sparse_module: eqx.Module, head_index: int) -> Callable[[jnp.ndarray, Optional[object]], jnp.ndarray]:
        """
        Create a hook that replaces activations of a specific attention head 
        with SAE reconstructions.
        """
        def hook(activations: jnp.ndarray) -> jnp.ndarray:
            # activations shape: [batch, seq_len, n_heads, d_head]
            head_slice = activations[:, :, head_index]
            sae_out, *_ = sparse_module(head_slice)
            return activations.at[:, :, head_index].set(sae_out.astype(activations.dtype))

        return hook

    # === Ablation Hooks ===

    @staticmethod
    def zero_ablate_hook(activations: jnp.ndarray) -> jnp.ndarray:
        """
        Replace all activations with zeros.
        """
        return jnp.zeros_like(activations)

    @staticmethod
    def mean_ablate_hook(activations: jnp.ndarray) -> jnp.ndarray:
        """
        Replace activations with the global mean value across batch and sequence dimensions.
        """
        mean_val = jnp.mean(activations, axis=(0, 1), keepdims=True)
        return jnp.broadcast_to(mean_val, activations.shape).astype(activations.dtype)
    
    # === Model Hook Attachment ===

    @staticmethod
    def with_node_hook(
        llm: eqx.Module, 
        layer_idx: int, 
        hook_attr: str, 
        fn: Callable[[jnp.ndarray, Optional[object]], jnp.ndarray]
    ) -> eqx.Module:
        """
        Attach a hook to a specific node of a given model layer.
        Returns a new model with the hook function applied at the given node.
        """
        def _get_hook_node(module: eqx.Module):
            try:
                return getattr(module.base_module.model.base_module.layers[layer_idx].base_module, hook_attr) #TODO: this is very closely taylored to the hook names of LLama ... make more general
            except AttributeError: # TODO also this exception is more of a hack - i just wanted a quick solution here
                return getattr(module.model.base_module.layers[layer_idx].base_module, hook_attr)

        return eqx.tree_at(lambda m: _get_hook_node(m).hook_function, llm, fn)