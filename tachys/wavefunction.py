import functools
from typing import Any, Callable

import jax
import jax.numpy as jnp
from flax import struct
from jax.flatten_util import ravel_pytree

from tachys.lattice.state_array import get_array


def _is_single_configuration(state):
    """True for a State holding one configuration, without the batch axis."""
    try:
        return get_array(state).ndim == 1
    except TypeError:       # a State that get_array cannot read: pass it on as is
        return False


def _with_batch_axis(apply_fn):
    """Wrap apply_fn so that it always receives a batch of configurations.

    A single configuration -- the optimizers evaluate the ansatz on one at a
    time for the per-sample Jacobians -- gets a leading axis of size one on
    every data leaf, and the result loses it again. Every leaf, not only the
    physical array: a foundation state's per-sample ``system_ids`` is 0-d, and
    ``jnp.atleast_2d`` would get it, and the batched (N_mc,) one, wrong.

    Wrapping is idempotent. ``__post_init__`` runs again on every ``replace``,
    ``jax.tree.map`` and jit trace, and a new wrapper each time would change the
    static pytree structure, so every jitted function would recompile.
    """
    if getattr(apply_fn, "_adds_batch_axis", False):
        return apply_fn

    @functools.wraps(apply_fn)
    def batched_apply_fn(params, state, *args, **kwargs):
        if _is_single_configuration(state):
            batch = jax.tree.map(lambda x: x[None], state)
            return apply_fn(params, batch, *args, **kwargs)[0]
        return apply_fn(params, state, *args, **kwargs)

    batched_apply_fn._adds_batch_axis = True
    return batched_apply_fn


class WaveFunction(struct.PyTreeNode):
    """Parameters paired with the function that evaluates them:
    ``apply_fn(params, state)`` returns log psi(x), with psi(x) = <x|psi>, for
    every configuration x of the batch. The full conventions -- basis
    ordering, what operators return, what ``compute_expectation`` computes --
    are stated in ``tachys.lattice.operator.local_estimator``.

    The ansatz always receives a batch. ``apply_fn`` is wrapped so that a
    single configuration, without the batch axis, is evaluated as a batch of
    one and returns a single log-amplitude; modules therefore never need to
    handle unbatched input. ``model.init`` is not wrapped: call it with a batch.
    The original function is ``wf.apply_fn.__wrapped__``.
    """
    params: Any
    apply_fn: Callable = struct.field(pytree_node=False)
    unravel_params_fn: Callable = struct.field(pytree_node=False, default=None)
    dtype: Any = struct.field(pytree_node=False, default=jnp.float64)

    def __post_init__(self):
        object.__setattr__(self, 'apply_fn', _with_batch_axis(self.apply_fn))
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
