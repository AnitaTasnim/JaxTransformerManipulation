import jax.numpy as jnp

def make_llama_inputs(batch_tokens: jnp.ndarray, pad_id: int = 0):
        """
        Build the full set of inputs (input_ids, attention_mask, position_ids)
        from bare batch_tokens. This keeps SAE code clean.
        """
        pad_id = 0 if pad_id is None else pad_id
        attention_mask = (batch_tokens != 0).astype(jnp.int32)
        seq_len = batch_tokens.shape[1]
        position_ids = jnp.broadcast_to(
            jnp.arange(seq_len, dtype=jnp.int32), batch_tokens.shape
        )
        return batch_tokens, attention_mask, position_ids