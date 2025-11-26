from typing import Tuple
import equinox as eqx
import jax
import jax.numpy as jnp
from equibox.models.llama import LlamaForCausalLM
from equibox_manipulation.hooks.hooks import hooked
from equibox_manipulation.transcoder.sparse_module_training.activations_store.ActivationsStore import ActivationsStore
from equibox_manipulation.transcoder.sparse_module_training.config import LanguageModelSAERunnerConfig
from equibox_manipulation.transcoder.sparse_module_training.transcoder_module.sparse_autoencoder import SparseAutoencoder
import os


class LMSparseAutoencoderSessionloader:
    """
    Responsible for loading all required
    artifacts and files for training
    a sparse autoencoder on a language model
    or analysing a pretraining autoencoder
    """

    def __init__(self, cfg: LanguageModelSAERunnerConfig):
        self.cfg = cfg

    def load_session(self) -> Tuple[eqx.Module, SparseAutoencoder, ActivationsStore]:
        """
        Loads a session for training a sparse autoencoder on a language model.
        """

        model = self.get_model()
        sparse_autoencoder = self.initialize_sparse_autoencoder()

        if self.cfg.lazy_device_loading:
            # In PyTorch this was done via monkey-patching with MethodType.
            # It offloaded competing modules when they were used to free up GPU memory.
            # In JAX/Equinox we cannot mutate modules like this - they live on one devce as they are immutable.
            # Instead, we use a lightweight move_to_device function
            model = self.move_to_device(model, jax.devices(self.cfg.model_device)[0])
            sparse_autoencoder = self.move_to_device(sparse_autoencoder, jax.devices(self.cfg.model_device)[0])


        activations_loader = self.get_activations_loader(model)

        return model, sparse_autoencoder, activations_loader
    
    @classmethod
    def load_session_from_pretrained(cls, path: str) -> Tuple[eqx.Module, SparseAutoencoder, ActivationsStore]:
        """
        Loads a session from a pretrained sparse autoencoder checkpoint.
        """
        raise NotImplementedError("Pretrained session loading must be implemented for JAX/Equinox.")


    def get_model(self) -> eqx.Module:
        """
        Loads a hooked Llama model using our Equibox implementation.
        Mimics TransformerLens API (HookedTransformer.from_pretrained).
        """

        model = LlamaForCausalLM.from_pretrained(self.cfg.model_name)
        model = hooked(model)

        # If cfg.model_dtype is set, cast all arrays
        if self.cfg.model_dtype is not None:
            model = jax.tree_util.tree_map(
                lambda x: x.astype(self.cfg.model_dtype) if isinstance(x, jnp.ndarray) else x,
                model,
            )

        # Optional: move params to device if specified
        if self.cfg.model_device is not None:
            device = jax.devices(self.cfg.model_device)[0]
            model = jax.tree_util.tree_map(
                lambda x: jax.device_put(x, device) if isinstance(x, jnp.ndarray) else x,
                model,
            )


        return model

    def initialize_sparse_autoencoder(self):
        key = jax.random.PRNGKey(self.cfg.seed)
        key, sae_key = jax.random.split(key) #TODO: pass fromm config
        sparse_autoencoder = SparseAutoencoder(self.cfg, key=key)
        return sparse_autoencoder

    def get_activations_loader(self, model) -> ActivationsStore:
        """
        Loads an ActivationsStore for the activations of a language model.

        In JAX/Equinox, `model` should already be wrapped with `hooked()`,
        so ActivationsStore can call into it and collect activations.
        """
        tokenizer = self.cfg.tokenizer
        activations_loader = ActivationsStore(self.cfg, model, tokenizer)
        return activations_loader


    def shuffle_activations_pairwise(datapath: str, buffer_idx_range: Tuple[int, int], key: jax.random.PRNGKey):
        """
        Shuffles two activation buffers on disk (JAX/Equinox version).
        """
        assert buffer_idx_range[0] < buffer_idx_range[1]

        buffer_idx1 = int(jax.random.randint(key, (1,), buffer_idx_range[0], buffer_idx_range[1])[0])
        key, subkey = jax.random.split(key)
        buffer_idx2 = int(jax.random.randint(subkey, (1,), buffer_idx_range[0], buffer_idx_range[1])[0])
        while buffer_idx1 == buffer_idx2:
            key, subkey = jax.random.split(key)
            buffer_idx2 = int(jax.random.randint(subkey, (1,), buffer_idx_range[0], buffer_idx_range[1])[0])

        path1 = os.path.join(datapath, f"{buffer_idx1}.eqx")
        path2 = os.path.join(datapath, f"{buffer_idx2}.eqx")
        buffer1 = eqx.tree_deserialise_leaves(path1, None)
        buffer2 = eqx.tree_deserialise_leaves(path2, None)

        joint_buffer = jnp.concatenate([buffer1, buffer2], axis=0)

        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, joint_buffer.shape[0])
        joint_buffer = joint_buffer[perm]

        shuffled_buffer1 = joint_buffer[: buffer1.shape[0]]
        shuffled_buffer2 = joint_buffer[buffer1.shape[0] :]

        eqx.tree_serialise_leaves(path1, shuffled_buffer1)
        eqx.tree_serialise_leaves(path2, shuffled_buffer2)

    def move_to_device(module, device):
        return jax.tree_util.tree_map(
            lambda x: jax.device_put(x, device) if isinstance(x, jax.Array) else x,
            module,
    )


