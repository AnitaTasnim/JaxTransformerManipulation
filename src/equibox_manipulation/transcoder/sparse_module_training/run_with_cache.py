from typing import Tuple
import jax
import jax.numpy as jnp
import equinox as eqx

def run_with_cache(
    llm,
    input_ids: jnp.ndarray,
    attention_mask: jnp.ndarray,
    position_ids: jnp.ndarray,
    layer_idx: int,
    hook_attr: str,
):
    captured = {}

    def capture_fn(x):
        captured["acts"] = x
        return x

    # robust accessor to the HookedModule node (works whether the root is HookedModule or plain)
    def _node(m):
        try:
            # root already HookedModule → go through .base_module to avoid __getattr__
            return getattr(m.base_module.model.base_module.layers[layer_idx].base_module, hook_attr)
        except AttributeError:
            # root plain eqx.Module → .model is already HookedModule
            return getattr(m.model.base_module.layers[layer_idx].base_module, hook_attr)

    # eqx.tree_at to set ONLY this node's hook_function = capture_fn
    llm_hooked = eqx.tree_at(lambda m: _node(m).hook_function, llm, capture_fn)

    # forward pass
    logits = llm_hooked(input_ids, attention_mask, position_ids)  # -> (B, T, V)
    ce = _per_token_ce(logits, input_ids)                          # -> (B, T)

    # read back activations stored by HookedModule.__call__()
    acts = _node(llm_hooked).activation                            # typically (B, T, d_in)
    if acts is None:
        raise RuntimeError(f"[ERROR] Hook at layer {layer_idx}.{hook_attr} did not fire.")

    return ce, acts


def _per_token_ce(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
        logp = jax.nn.log_softmax(logits, axis=-1)
        return -jnp.take_along_axis(logp, targets[..., None], axis=-1).squeeze(-1)  # (B, T)