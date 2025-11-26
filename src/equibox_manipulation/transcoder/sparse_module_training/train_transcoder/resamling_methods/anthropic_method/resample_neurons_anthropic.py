import jax
import jax.numpy as jnp
import equinox as eqx


def resample_neurons_anthropic(self, dead_neuron_indices, llm, opt_state, activation_store, key):
    """
    JAX/EQX version of Anthropic resampling.
    Returns new_sae, new_opt_state.
    """

    # 1) collect global loss increases + activations
    global_loss_increases, global_input_activations = self.collect_anthropic_resampling_losses(
        llm, activation_store, key
    )

    # 2) sample indices proportional to global loss increases
    probs = global_loss_increases / jnp.sum(global_loss_increases)
    n_dead = dead_neuron_indices.shape[0]
    n_samples = min(n_dead, probs.shape[0])

    sample_indices = jax.random.choice(
        key, probs.shape[0], shape=(n_samples,), replace=False, p=probs
    )

    if n_samples < n_dead:
        dead_neuron_indices = dead_neuron_indices[:n_samples]

    # 3) reinitialize W_dec for dead neurons
    sampled_acts = global_input_activations[sample_indices]
    sampled_acts = sampled_acts / (jnp.linalg.norm(sampled_acts, axis=1, keepdims=True) + 1e-8)

    W_dec_new = self.W_dec.at[dead_neuron_indices, :].set(sampled_acts)

    # 4) reinitialize W_enc + biases
    W_enc_new = self.W_enc.at[:, dead_neuron_indices].set(W_dec_new[dead_neuron_indices, :].T)
    b_enc_new = self.b_enc.at[dead_neuron_indices].set(0.0)

    if n_samples < self.d_sae:
        sum_norms = jnp.sum(jnp.linalg.norm(W_enc_new, axis=0)) - n_samples
        average_norm = sum_norms / (self.d_sae - n_samples)
        W_enc_new = W_enc_new.at[:, dead_neuron_indices].mul(self.cfg.feature_reinit_scale * average_norm)
        # biases reset factor
        relevant_biases = jnp.mean(b_enc_new[dead_neuron_indices])
        b_enc_new = b_enc_new.at[dead_neuron_indices].set(relevant_biases * 0.0)
    else:
        W_enc_new = W_enc_new.at[:, dead_neuron_indices].mul(self.cfg.feature_reinit_scale)
        b_enc_new = b_enc_new.at[dead_neuron_indices].set(-5.0)

    new_sae = eqx.tree_at(
        lambda m: (m.W_dec, m.W_enc, m.b_enc),
        self,
        (W_dec_new, W_enc_new, b_enc_new),
    )

    # 5) reset optimizer state
    # In JAX, re-init the optimizer instead of mutating:
    new_opt_state = opt_state  # placeholder
    # if using optax:
    # new_opt_state = optimizer.init(eqx.filter(new_sae, eqx.is_array))

    return new_sae, new_opt_state

def collect_anthropic_resampling_losses(self, llm, activation_store, key):
    """
    JAX/EQX version of the Anthropic resampling loss collection.
    - uses get_test_loss(...) for CE with SAE (transcoder path)
    - uses run_with_cache_jax(...) for baseline CE + activations at hook point
    Returns:
    global_loss_increases: (N_final,)
    global_input_activations: (N_final, d_in)
    """

    batch_size = int(self.cfg.store_batch_size)
    n_batches = int(self.cfg.resample_batches)
    n_final = batch_size * n_batches

    global_loss_increases = jnp.zeros((n_final,), dtype=self.dtype)
    global_input_activations = jnp.zeros((n_final, self.d_in), dtype=self.dtype)

    layer_idx = int(self.cfg.hook_point_layer)
    hook_attr = str(self.cfg.hook_point)
    head_idx = getattr(self.cfg, "hook_point_head_index", None)

    write_ptr = 0
    for b in range(n_batches):
        # 1) tokens → full llama inputs
        batch_tokens = activation_store.get_batch_tokens()                # (B, T)
        input_ids, attention_mask, position_ids = self.make_llama_inputs(batch_tokens)

        # 2) CE with SAE (transcoder: replaces MLP at layer)
        ce_with = self.get_test_loss(input_ids, attention_mask, position_ids, llm)  # (B, T)

        # 3) Baseline CE + activations at hook point
        ce_without, normal_acts = self.run_with_cache_jax(
            llm, input_ids, attention_mask, position_ids, layer_idx, hook_attr
        )  # ce_without: (B, T), normal_acts: (B, T, d_in) (usually)

        # optional head slice if the hook returns per-head (B,T,H,d) or similar
        if head_idx is not None:
            normal_acts = normal_acts[:, :, int(head_idx)]

        # 4) Δ-loss and sampling distribution
        delta = ce_with - ce_without                                   # (B, T)
        probs = jax.nn.relu(delta)
        denom = jnp.sum(probs, axis=1, keepdims=True)
        # safe normalisation: if a row sums to 0, use uniform
        T = probs.shape[1]
        probs = jnp.where(
            denom > 0,
            probs / denom,
            jnp.full_like(probs, 1.0 / T),
        )

        # 5) sample one token position per sequence
        subkeys = jax.random.split(key, batch_size)
        samples = jax.vmap(lambda k, p: jax.random.choice(k, p.shape[0], p=p))(subkeys, probs)  # (B,)
        b_idx = jnp.arange(batch_size)

        picked_loss = delta[b_idx, samples]          # (B,)
        picked_acts = normal_acts[b_idx, samples]    # (B, d_in)

        # 6) write into global buffers
        end = write_ptr + batch_size
        global_loss_increases = global_loss_increases.at[write_ptr:end].set(picked_loss.astype(self.dtype))
        global_input_activations = global_input_activations.at[write_ptr:end].set(picked_acts.astype(self.dtype))
        write_ptr = end

        # advance rng
        key = jax.random.fold_in(key, b)

        # tiny debug
        print(f"[collect] batch {b+1}/{n_batches}: ΔCE mean={jnp.mean(picked_loss):.4f}, acts={picked_acts.shape}")

    return global_loss_increases, global_input_activations