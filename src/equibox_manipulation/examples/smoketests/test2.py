import jax.numpy as jnp

# ------------------------------------------------------------
# Fake Tokenizer
# ------------------------------------------------------------
class DummyTokenizer:
    def __init__(self):
        self.vocab = {"<pad>": 0, "<bos>": 1}
        self.bos_token_id = 1

    def __call__(self, text, truncation=False, return_tensors=None):
        # Ganz simpler Tokenizer: jede Figur = ID
        ids = [self.vocab.get("<bos>")]
        ids.extend([ord(c) % 100 for c in text])  # einfache Hashung
        return {"input_ids": jnp.array([ids], dtype=jnp.int32)}

# ------------------------------------------------------------
# Dummy Config
# ------------------------------------------------------------
class DummyCfg:
    store_batch_size = 3
    context_size = 6
    data_column = "text"
    is_dataset_tokenized = False

# ------------------------------------------------------------
# Dummy Dataset
# ------------------------------------------------------------
class DummyDataset:
    def __init__(self, texts):
        self.texts = texts
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.i >= len(self.texts):
            self.i = 0
        item = {"text": self.texts[self.i]}
        self.i += 1
        return item

# ------------------------------------------------------------
# ActivationsStore mit get_batch_tokens
# ------------------------------------------------------------
class ActivationsStore:
    def __init__(self, cfg, tokenizer, dataset):
        self.cfg = cfg
        self.model = type("DummyModel", (), {"tokenizer": tokenizer})()
        self.iterable_dataset = iter(dataset)

    def get_batch_tokens(self):
        batch_size = self.cfg.store_batch_size
        context_size = self.cfg.context_size
        bos_token_id = self.model.tokenizer.bos_token_id

        batch_tokens = jnp.zeros((batch_size, context_size), dtype=jnp.int32)
        filled = 0

        while filled < batch_size:
            sample = next(self.iterable_dataset)[self.cfg.data_column]
            enc = self.model.tokenizer(sample, truncation=False, return_tensors="np")
            tokens = jnp.array(enc["input_ids"].squeeze(0), dtype=jnp.int32)

            token_len = tokens.shape[0]
            current_tokens = jnp.zeros((0,), dtype=jnp.int32)
            current_length = 0

            while token_len > 0 and filled < batch_size:
                space_left = context_size - current_length
                if token_len <= space_left:
                    current_tokens = jnp.concatenate([current_tokens, tokens[:token_len]])
                    current_length += token_len
                    break
                else:
                    current_tokens = jnp.concatenate([current_tokens, tokens[:space_left]])
                    tokens = jnp.concatenate([jnp.array([bos_token_id], dtype=jnp.int32),
                                              tokens[space_left:]])
                    token_len = tokens.shape[0]
                    current_length = context_size

                if current_length == context_size:
                    batch_tokens = batch_tokens.at[filled].set(current_tokens)
                    filled += 1
                    current_tokens = jnp.zeros((0,), dtype=jnp.int32)
                    current_length = 0

            if current_length > 0 and filled < batch_size:
                pad_len = context_size - current_length
                padded = jnp.concatenate([current_tokens, jnp.zeros((pad_len,), dtype=jnp.int32)])
                batch_tokens = batch_tokens.at[filled].set(padded)
                filled += 1

        return batch_tokens

# ------------------------------------------------------------
# Testlauf
# ------------------------------------------------------------
if __name__ == "__main__":
    cfg = DummyCfg()
    tokenizer = DummyTokenizer()
    dataset = DummyDataset(["Hi", "Heidelberg", "JAX Rocks!"])
    store = ActivationsStore(cfg, tokenizer, dataset)

    batch = store.get_batch_tokens()
    print("Batch shape:", batch.shape)
    print(batch)
