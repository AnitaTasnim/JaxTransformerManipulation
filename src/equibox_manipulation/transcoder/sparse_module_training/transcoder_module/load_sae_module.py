import os
import pickle
import gzip
import equinox as eqx


def load_sae_model(path: str, sae_template=None):
    """
    Load SparseAutoencoder from disk.
    If .eqx: needs a template SAE instance with same structure.
    If .pkl.gz: restores config + model.
    """
    if path.endswith(".eqx"):
        if sae_template is None:
            raise ValueError("sae_template must be provided when loading .eqx")
        return eqx.tree_deserialise_leaves(path, sae_template)

    elif path.endswith(".pkl.gz"):
        with gzip.open(path, "rb") as f:
            state = pickle.load(f)
        return state["model"], state["cfg"]

    else:
        raise ValueError(f"Unexpected extension: {path}")
