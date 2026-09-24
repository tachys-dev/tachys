from typing import Any, Callable

import jax
import jax.numpy as jnp
from flax import struct
from jax.flatten_util import ravel_pytree


class WaveFunction(struct.PyTreeNode):
    """Parameters paired with the function that evaluates them:
    ``apply_fn(params, state)`` returns log psi(x), with psi(x) = <x|psi>, for
    every configuration x of the batch. The full conventions -- basis
    ordering, what operators return, what ``compute_expectation`` computes --
    are stated in ``tachys.lattice.operator.local_estimator``.
    """
    params: Any
    apply_fn: Callable = struct.field(pytree_node=False)
    unravel_params_fn: Callable = struct.field(pytree_node=False, default=None)
    dtype: Any = struct.field(pytree_node=False, default=jnp.float64)

    def __post_init__(self):
        if self.unravel_params_fn is None:
            _, unravel_fn = ravel_pytree(self.params)
            object.__setattr__(self, 'unravel_params_fn', unravel_fn)

    @jax.jit
    def apply_gradients(self, grads, eta):
        new_params = jax.tree.map(lambda p, g: p - eta * g, self.params, grads)
        return self.replace(params=new_params)

    @property
    def num_params(self):
        flat, _ = ravel_pytree(self.params)
        return flat.size
