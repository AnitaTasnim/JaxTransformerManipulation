import copy
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

import equinox as eqx
import jax.numpy as jnp
import jax

# This code currently makes a few bold assumptions, namely:
#
# (i) All `eqx.Module`s implement a method `__call__`
#     (this is not necessarily true, see https://docs.kidger.site/equinox/api/module/module/)
# (ii) None of the functions are optimized using @jax.jit (or similar)


T = TypeVar("T")


@dataclass
class Mutable(Generic[T]):
    value: T | None = None


class HookedModule(eqx.Module):
    base_module: eqx.Module
    hook_function: Callable

    # Ugly hack: Because modules are frozen but we need to save some state, we introduce an instance that has
    # mutable state. While the reference to this instance cannot be changed, the properties of the instance can
    # be changed. This way, we get mutable state in an immutable class.
    __activation_wrapper: Mutable[jnp.array] = field(
        init=False, default_factory=Mutable
    )

    def __call__(self, *args, **kwargs):
        """
        A hooked forward pass.

        It will pass on all arguments (positional and keyword) to the base module.
        """
        # Run a normal forward pass, by calling the base module with whatever arguments we were given.
        # This only works because of assumption (i)
        base_activation = self.base_module(*args, **kwargs)

        # As we are also doing write-hooking, we now apply our own hook function that can take the activations
        # and modify them in some way
        potentially_modified_activation = self.hook_function(base_activation)

        # We now store the potentially modified activation
        self.__activation_wrapper.value = potentially_modified_activation

        return potentially_modified_activation

    @property
    def activation(self) -> jnp.array:
        # Convenience property to make it easier to access the current activation
        return self.__activation_wrapper.value

    def __getattr__(self, name: str) -> Any:
        # Convenience wrapper: If one tries to access a property on the hooked module that does not exist,
        # this method is called. We then delegate this to self.base_module.
        return getattr(self.base_module, name)


def hooked(
    module: eqx.Module,
    get_hook_function: Callable[[eqx.Module], Callable | None] = lambda _: None,
) -> eqx.Module:
    cloned_module = jax.tree_util.tree_map(lambda x: x, module) #copy.copy(module)
    # copy.copy(module) triggert bei deinen (Equinox-)Modulen über __getattr__ eine Endlos-Kaskade → RecursionError. Das passiert, weil dein HookedModule.__getattr__ alles an base_module weiterreicht und copy.copy beim Shallow-Copy intern diverse Attribute abfragt. Ergebnis: rekursiver Attribute-Lookup.

    # Most PyTree manipulation methods operate on the leaves of a subtree, not the
    # subtrees themselves (e.g. jax.tree.map).
    #
    # While it is possible to replace subtrees using eqx.tree_at, we would still need
    # to write a "where" function that, returns all subtrees (but not leaves) of a PyTree,
    # given the PyTree. To do this dynamically, one would have to use getattr()/setattr()
    # or __dict__.
    #
    # We therefore decided to skip the eqx.tree_at method, and directly operate on the
    # __dict__ property of the module.

    for obj_name, obj in cloned_module.__dict__.items():
        if isinstance(obj, eqx.Module):
            cloned_module.__dict__[obj_name] = hooked(obj, get_hook_function)
        if isinstance(obj, list):
            # This only handles one possible sequence type, i.e., list. Support for tuples
            # and other sequences can be added in a similar fashion.
            cloned_module.__dict__[obj_name] = [
                hooked(el, get_hook_function) if isinstance(el, eqx.Module) else el
                for el in obj
            ]
    #print(f"[DEBUG] Setting hook_function on {obj_name} to {get_hook_function(obj)}")

    return HookedModule(
        cloned_module, hook_function=get_hook_function(module) or (lambda x: x)
    )