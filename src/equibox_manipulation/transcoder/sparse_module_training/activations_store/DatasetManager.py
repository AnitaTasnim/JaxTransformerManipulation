from datasets import load_dataset
import jax.numpy as jnp
import jax
from equibox_manipulation.transcoder.sparse_module_training.utils import jax_key_to_seed

class DatasetManager:

    def __init__(self, cfg, tokenizer, key):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.key, subkey = jax.random.split(key)
        seed = jax_key_to_seed(subkey)

        self.dataset = load_dataset(cfg.dataset_path, split="train", streaming=True).shuffle(seed=seed)
        self.iterable_dataset = iter(self.dataset)

        # infer tokenization state if not provided
        if cfg.is_dataset_tokenized is None:
            first_item = next(self.iterable_dataset)[cfg.data_column]
            cfg.is_dataset_tokenized = not isinstance(first_item, str)

    

    def get_batch_tokens(self):
        """
        Streams a batch of tokens from the dataset, tokenizes them, splits into contexts,
        and returns a padded batch of shape (batch_size, context_size).
        """
        batch_size = self.cfg.store_batch_size
        context_size = self.cfg.context_size
        bos_token_id = self.tokenizer.bos_token_id or self.tokenizer.eos_token

        batch_tokens = jnp.zeros((batch_size, context_size), dtype=jnp.int32)
        filled = 0

        while filled < batch_size:
            sample = self._get_next_sample()
            tokens = self._tokenize_sample(sample)
            contexts = self._split_into_contexts(tokens, context_size, bos_token_id)

            for context in contexts:
                if filled >= batch_size:
                    break
                batch_tokens = batch_tokens.at[filled].set(
                    self._pad_to_context(context, context_size)
                )
                filled += 1

        return batch_tokens

    def _get_next_sample(self):
        """Fetch next raw sample from dataset."""
        return next(self.iterable_dataset)[self.cfg.data_column]

    def _tokenize_sample(self, sample):
        """Convert sample (str or already tokenized) into token IDs."""
        if not self.cfg.is_dataset_tokenized:
            enc = self.tokenizer(
                sample,
                max_length=self.cfg.context_size,
                truncation=True,
                padding="max_length", #TODO check impact of this - torch implement didnt cut to context length but i had memory issues without it
                return_tensors="np",
            )
            return jnp.array(enc["input_ids"].squeeze(0), dtype=jnp.int32)
        else:
            return jnp.array(sample, dtype=jnp.int32)

    def _split_into_contexts(self, tokens, context_size, bos_token_id):
        """
        Yield token sequences of length <= context_size.
        Handles splitting across multiple contexts and reinserting bos_token.
        """
        contexts = []
        current_tokens = jnp.zeros((0,), dtype=jnp.int32)

        while tokens.shape[0] > 0:
            space_left = context_size - current_tokens.shape[0]

            if tokens.shape[0] <= space_left:
                current_tokens = jnp.concatenate([current_tokens, tokens])
                contexts.append(current_tokens)
                break
            else:
                current_tokens = jnp.concatenate([current_tokens, tokens[:space_left]])
                contexts.append(current_tokens)
                tokens = jnp.concatenate([jnp.array([bos_token_id], dtype=jnp.int32), tokens[space_left:]])
                current_tokens = jnp.zeros((0,), dtype=jnp.int32)

        return contexts

    def _pad_to_context(self, tokens, context_size):
        """Right-pad token sequence with zeros to match context_size."""
        pad_len = context_size - tokens.shape[0]
        if pad_len > 0:
            return jnp.concatenate([tokens, jnp.zeros((pad_len,), dtype=jnp.int32)])
        return tokens
