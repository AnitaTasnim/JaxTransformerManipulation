from typing import Any, Optional
import equinox as eqx
import jax, jax.numpy as jnp
from equibox_manipulation.transcoder.sparse_module_training.transcoder_module.GhostGradHandler import GhostGradHandler


class SparseAutoencoder(eqx.Module):
    cfg: Any = eqx.field(static=True)
    d_in: int = eqx.field(static=True)
    d_sae: int = eqx.field(static=True)
    d_out: int = eqx.field(static=True)
    l1_coefficient: float = eqx.field(static=True)
    dtype: jnp.dtype = eqx.field(static=True)

    W_enc: jax.Array
    b_enc: jax.Array
    W_dec: jax.Array
    b_dec: jax.Array
    b_dec_out: Optional[jax.Array]

    spacon_sae_W_dec: Optional[jax.Array] = None

    ghost_grad_handler: GhostGradHandler = eqx.field(static=True)

    def __init__(self, cfg, key: jax.Array):
        self.cfg = cfg
        self.d_in = int(cfg.d_in)
        self.d_sae = int(cfg.d_sae)
        self.l1_coefficient = float(cfg.l1_coefficient)
        self.dtype = cfg.dtype
        self.spacon_sae_W_dec = None

        self.d_out = self.d_in if not (cfg.is_transcoder and cfg.d_out is not None) else int(cfg.d_out)

        k1, k2 = jax.random.split(key)
        init = jax.nn.initializers.variance_scaling(2.0, "fan_in", "uniform")
        self.W_enc = init(k1, (self.d_in, self.d_sae), self.dtype)
        self.b_enc = jnp.zeros((self.d_sae,), dtype=self.dtype)

        W_dec = init(k2, (self.d_sae, self.d_out), self.dtype)
        self.W_dec = W_dec / jnp.linalg.norm(W_dec, axis=1, keepdims=True)

        self.b_dec = jnp.zeros((self.d_in,), dtype=self.dtype)
        self.b_dec_out = jnp.zeros((self.d_out,), dtype=self.dtype) if cfg.is_transcoder else None

        self.ghost_grad_handler = GhostGradHandler(dtype=self.dtype)

    def __call__(self, x, dead_neuron_mask=None, mse_target=None, training: bool = True):
        # 1) match dtype
        x = x.astype(self.dtype)
        if isinstance(x, jax.Array):
            device = jax.devices(self.cfg.device)[0]
            x = jax.device_put(x, device) #TODO ugly hack ... why isnt it on correct device already? trace back and fix!

        # 2) subtract decoder bias and hook
        sae_in = x - self.b_dec
        sae_in = self.hook_sae_in(sae_in)

        # 3) encode + bias + hook
        hidden_pre = jnp.einsum("...i,ij->...j", sae_in, self.W_enc) + self.b_enc
        hidden_pre = self.hook_hidden_pre(hidden_pre)

        # 4) ReLU
        feature_acts = jax.nn.relu(hidden_pre)

        # 5) optional top-k sparsification
        if self.cfg.top_k is not None:
            flat = feature_acts.reshape(-1)
            k = int(min(self.cfg.top_k * x.shape[0], flat.size))
            top_vals, top_idx = jax.lax.top_k(flat, k)
            mask = jnp.zeros_like(flat)
            mask = mask.at[top_idx].set(top_vals)
            feature_acts = mask.reshape(feature_acts.shape)

        # 6) hook after sparsification
        feature_acts = self.hook_hidden_post(feature_acts)

        # 7) decode
        if self.cfg.is_transcoder:
            sae_out = jnp.einsum("...i,ij->...j", feature_acts, self.W_dec) + self.b_dec_out
        else:
            sae_out = jnp.einsum("...i,ij->...j", feature_acts, self.W_dec) + self.b_dec
        sae_out = self.hook_sae_out(sae_out)

        if not training:
            # Inference mode: just return reconstruction & activations
            return sae_out, feature_acts, None, None, None, None

        # -------- TRAINING MODE BELOW --------
        # 8) reconstruction loss (normalized MSE)
        target = x if mse_target is None else mse_target.astype(jnp.float32)
        denom = jnp.sqrt(jnp.sum(target**2, axis=-1, keepdims=True) + 1e-12)
        mse_loss = ((sae_out - target) ** 2) / denom

        mse_loss_ghost_resid = jnp.array(0.0, dtype=self.dtype)

        # 9) ghost grads protocol
        #if self.cfg.use_ghost_grads and dead_neuron_mask is not None and jnp.sum(dead_neuron_mask) > 0:
        
        def with_ghostgradients(_):
            return self.ghost_grad_handler.with_ghostgradients(x=x, sae_out=sae_out, hidden_pre=hidden_pre, dead_neuron_mask=dead_neuron_mask, W_dec=self.W_dec, mse_loss=mse_loss)

        def without_ghostgradients(_):
            return self.ghost_grad_handler.without_ghostgradients(_)
        
        if self.cfg.use_ghost_grads and dead_neuron_mask is not None:
            mse_loss_ghost_resid = jax.lax.cond(
                jnp.any(dead_neuron_mask),
                with_ghostgradients,
                without_ghostgradients,
                operand=(),
            )
        else:
            mse_loss_ghost_resid = jnp.array(0.0, dtype=self.dtype)

        mse_loss = jnp.mean(mse_loss)

        # 11) sparsity penalty
        sparsity = jnp.mean(jnp.sum(jnp.abs(feature_acts), axis=1))
        l1_loss = self.l1_coefficient * sparsity

        # 12) total loss
        loss = mse_loss + l1_loss + mse_loss_ghost_resid

        return sae_out, feature_acts, loss, mse_loss, l1_loss, mse_loss_ghost_resid

    
    # ---- Hooks (default: identity) ----
    def hook_sae_in(self, x): return x
    def hook_hidden_pre(self, x): return x
    def hook_hidden_post(self, x): return x
    def hook_sae_out(self, x): return x
