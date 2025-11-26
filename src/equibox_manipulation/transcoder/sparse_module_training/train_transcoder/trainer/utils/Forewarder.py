import jax.numpy as jnp
import equinox as eqx


class Forewarder:
    """Utility functions for training sparse modules."""


    @staticmethod
    def forward_train(sparse_module: eqx.Module, inputs: jnp.ndarray, dead_neuron_mask: jnp.ndarray, mse_target: jnp.ndarray):
        return sparse_module(inputs, dead_neuron_mask=dead_neuron_mask, mse_target=mse_target, training=True)

    @staticmethod
    def forward_infer(sparse_module: eqx.Module, inputs: jnp.ndarray):
        return sparse_module(inputs, training=False)

    
