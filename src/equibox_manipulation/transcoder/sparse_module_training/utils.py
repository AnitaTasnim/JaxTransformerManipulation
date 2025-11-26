import jax.numpy as jnp
import jax

def jax_key_to_seed(key: jax.Array) -> int:
        return int(jax.random.bits(key, dtype=jnp.uint32, shape=()).item())

def get_device(device_str: str):
    if device_str == "cpu":
        return jax.devices("cpu")[0]
    elif device_str == "gpu":
        return jax.devices("gpu")[0]
    else:
        raise ValueError(f"Unknown device string: {device_str}")