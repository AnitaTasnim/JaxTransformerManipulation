import jax
import jax.numpy as jnp
import equinox as eqx

def resample_neurons_l2(self, x, feature_sparsity, opt_state, key):
        """
        Resamples dead neurons in JAX/eqx.
        Returns new_module, new_opt_state, n_dead.
        """
        feature_reinit_scale = self.cfg.feature_reinit_scale

        sae_out, _, _, _, _, _ = self(x, training=False)
        per_token_l2_loss = jnp.sum((sae_out - x) ** 2, axis=-1)

        # 1) detect dead neurons
        is_dead = feature_sparsity < self.cfg.dead_feature_threshold
        dead_idx = jnp.nonzero(is_dead, size=is_dead.shape[0], fill_value=-1)[0]
        alive_idx = jnp.nonzero(~is_dead, size=is_dead.shape[0], fill_value=-1)[0]
        n_dead = int(jnp.sum(is_dead))

        if n_dead == 0 or jnp.max(per_token_l2_loss) < 1e-6:
            return self, opt_state, 0

        # 2) sample replacements proportional to squared loss
        probs = (per_token_l2_loss ** 2) / jnp.sum(per_token_l2_loss ** 2)
        replacement_indices = jax.random.choice(key, x.shape[0], shape=(n_dead,), p=probs)

        # 3) replacement values (centered, unit norm)
        replacement_values = (x - self.b_dec)[replacement_indices]  # (n_dead, d_in)
        replacement_values = replacement_values / (
            jnp.linalg.norm(replacement_values, axis=1, keepdims=True) + 1e-8
        )

        # 4) reinitialize decoder + encoder weights
        W_dec_new = self.W_dec.at[is_dead, :].set(replacement_values)

        if jnp.sum(~is_dead) > 0:
            W_enc_norm_alive_mean = jnp.mean(
                jnp.linalg.norm(self.W_enc[:, ~is_dead], axis=0)
            )
        else:
            W_enc_norm_alive_mean = 1.0

        W_enc_new = self.W_enc.at[:, is_dead].set(
            (replacement_values * W_enc_norm_alive_mean * feature_reinit_scale).T
        )

        b_enc_new = self.b_enc.at[is_dead].set(0.0)

        new_module = eqx.tree_at(
            lambda m: (m.W_dec, m.W_enc, m.b_enc),
            self,
            (W_dec_new, W_enc_new, b_enc_new),
        )

        # 5) reset optimizer state
        # easiest path in JAX: re-init optimizer state from scratch
        # you can also implement "zeroing out dead indices" if you want exact equivalence
        new_opt_state = opt_state  # placeholder
        # e.g. if you're using optax: new_opt_state = optimizer.init(eqx.filter(new_module, eqx.is_array))

        return new_module, new_opt_state, n_dead