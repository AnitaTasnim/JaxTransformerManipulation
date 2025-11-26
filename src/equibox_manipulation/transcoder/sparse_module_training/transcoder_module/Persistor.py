import equinox as eqx
import os
import json
from pathlib import Path
import wandb

class Persistor:
    @classmethod
    def save_sae_model(cls, cfg, sparse_module):
        """Speichert das Sparse Autoencoder Modell als .eqx Datei."""
        if wandb.run is not None:
            run_folder = Path(cfg.checkpoint_path) / wandb.run.id
        else:
            run_folder = Path(cfg.checkpoint_path)

        final_path = run_folder / f"final_{cls.get_name(cfg)}.eqx"
        os.makedirs(final_path.parent, exist_ok=True)

        print(f"[INFO] Saving SAE to: {final_path}")
        eqx.tree_serialise_leaves(final_path, sparse_module)
        return final_path
    
    def get_name(self, cfg):
        safe_model_name = cfg.model_name.replace("/", "_")
        sae_name = f"sparse_autoencoder_{safe_model_name}_{cfg.hook_point}_{cfg.d_sae}"
        return sae_name

    @classmethod
    def load_sae_model(path: str, cls, cfg, key): #TODO load from JaxTransformerHub instead
        """
        Load SAE + hyperparams.
        Returns (hyperparams, model).
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No file found at {path}")

        with open(path, "rb") as f:
            hyperparams = json.loads(f.readline().decode())
            # Build a template so eqx knows the structure
            template = cls(cfg, key)
            model = eqx.tree_deserialise_leaves(f, template)
        

        return hyperparams, model