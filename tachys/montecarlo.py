from functools import partial

import jax
import jax.numpy as jnp
from flax import struct
from jax.sharding import PartitionSpec as P
from jax.experimental.shard_map import shard_map

from tachys.parallel import mesh
from tachys.lattice.operator.local_estimator import _apply_masked
from tachys.utils import _cast_floating_to


class _BaseAction(struct.PyTreeNode):
    """Base class for MCMC move proposals.

    Subclasses must implement ``__call__(self, key, state)`` returning
    ``(new_state, allowed_move, log_prob_correction, action_id)`` where:

    - ``new_state``: proposed State after applying the move.
    - ``allowed_move``: boolean mask (N_mc,) — False when the move is a
      no-op (e.g. exchanging identical spins or identical occupations),
      allowing early rejection before evaluating the wavefunction.
    - ``log_prob_correction``: scalar log-probability correction for
      asymmetric proposals; 0.0 for symmetric moves.
    - ``action_id``: integer index of the sub-action used (scalar 0 for
      atomic actions; per-chain array for ``CompositeAction``).

    ``state`` can be any subclass of ``State``.

    Subclasses define ``__call__`` returning 3 values (atomic actions) or
    4 values (when a custom ``action_id`` is needed). ``__init_subclass__``
    automatically wraps any subclass ``__call__`` to append ``jnp.int32(0)``
    when only 3 values are returned.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if '__call__' in cls.__dict__:
            _orig = cls.__dict__['__call__']
            def _wrapped(self, key, state, _orig=_orig):
                result = _orig(self, key, state)
                return (*result, jnp.int32(0)) if len(result) == 3 else result
            cls.__call__ = _wrapped

    def __call__(self, *_):
        raise NotImplementedError

    @property
    def n_actions(self):
        """Number of distinct sub-actions (1 for atomic actions)."""
        return 1


class CompositeAction(_BaseAction):
    """Randomly selects among n actions on each chain independently.

    On each step, chain i applies ``actions[k]`` with probability ``probs[k]``.

    Attributes:
        actions: Tuple of actions to choose from.
        probs  : Tuple of probabilities, one per action. Must sum to 1.
    """

    actions: tuple
    probs: tuple = struct.field(pytree_node=False)

    def __post_init__(self):
        assert len(self.actions) == len(self.probs), "actions and probs must have the same length"
        assert abs(sum(self.probs) - 1.0) < 1e-6, "probs must sum to 1"

    @property
    def n_actions(self):
        return len(self.probs)

    def __call__(self, key, state):
        key, subkey = jax.vmap(jax.random.split, out_axes=1)(key)
        rand = jax.vmap(jax.random.uniform)(subkey)

        cumprobs  = jnp.cumsum(jnp.array(self.probs[:-1]))
        action_id = jnp.sum(rand[:, None] > cumprobs[None, :], axis=1).astype(int)

        def _select(action_id, key, state):
            key   = jnp.atleast_1d(key)
            state = jax.tree.map(jnp.atleast_2d, state)
            return jax.lax.switch(action_id, self.actions, key, state)

        new_state, allowed_move, log_prob_correction, _ = jax.vmap(_select)(action_id, key, state)

        allowed_move = allowed_move[:, 0]
        new_state    = jax.tree.map(lambda x: x[:, 0], new_state)

        return new_state, allowed_move, log_prob_correction, action_id


def mc_step(state, key, action, wf, log_amps, optimize_mask=True, batch_expand=1):
    """Perform one Metropolis–Hastings step across all chains.

    Parameters
    ----------
    state         : State, batch axis 0 of size N_mc.
    key           : jax.random.key array of shape (N_mc,), one key per chain.
    action        : _BaseAction — callable that proposes a move.
    wf            : WaveFunction with ``.apply_fn(params, state) -> (N_mc,)`` log-amplitudes.
    log_amps      : jax.Array, shape (N_mc,) — current log-amplitudes.
    optimize_mask : bool — skip wavefunction evaluation for trivially rejected moves.
    batch_expand  : int — batch enlargement factor passed to ``_apply_masked``.

    Returns
    -------
    state, key, log_amps, accepted, action_id
        ``accepted``  is a boolean array of shape (N_mc,).
        ``action_id`` is a scalar or per-chain int array identifying the sub-action used.
    """
    keys = jax.vmap(partial(jax.random.split, num=3))(key)
    key, subkey1, subkey2 = keys.T

    new_state, allowed_move, log_prob_correction, action_id = action(subkey1, state)

    if optimize_mask:
        _state = jax.tree.map(lambda x: x[None], new_state)
        log_amps_new = _apply_masked(wf, _state, allowed_move[None], log_amps, batch_expand=batch_expand)
        log_amps_new = log_amps_new[0]
    else:
        log_amps_new = wf.apply_fn(wf.params, new_state)

    log_prob = 2.0 * (jnp.real(log_amps_new) - jnp.real(log_amps)) + log_prob_correction

    rands = 1.0 - jax.vmap(jax.random.uniform)(subkey2)
    accepted = (jnp.log(rands) < log_prob) * allowed_move

    state    = jax.tree.map(lambda x, y: jnp.where(accepted[:, None], x, y), new_state, state)
    log_amps = jnp.where(accepted, log_amps_new, log_amps)

    return state, key, log_amps, accepted, action_id


@jax.jit
@partial(shard_map,
         mesh=mesh,
         in_specs=(P(), P('i'), P(None), P('i'), P(None)),
         out_specs=(P('i'), P('i'), P()),
         check_rep=False,
         )
def sample(nsweeps, state, action, key, wf):
    """Run ``nsweeps * Ns`` Metropolis steps across all sharded chains.

    Parameters
    ----------
    nsweeps : int — number of sweeps (one sweep = Ns steps).
    state   : State, batch axis 0 of size N_mc.
    action  : _BaseAction.
    key     : jax.random.key array of shape (N_mc,).
    wf      : WaveFunction.

    Returns
    -------
    state, log_amps, acceptance
        ``acceptance`` is a jnp.array of shape ``(n_actions,)`` with the
        per-action acceptance rate (accepted / selected) averaged globally.
    """
    wf, state = _cast_floating_to((wf, state), wf.dtype)

    log_amps   = wf.apply_fn(wf.params, state)
    Ns         = state.Ns
    N_mc_local = state.spins.shape[0]
    n_actions  = action.n_actions

    acc_sum = jnp.zeros((n_actions, N_mc_local))
    sel_sum = jnp.zeros((n_actions, N_mc_local))

    def _step(_, vals):
        state, key, log_amps, acc_sum, sel_sum = vals
        state, key, log_amps, accepted, action_id = mc_step(state, key, action, wf, log_amps)
        chain_idx = jnp.arange(N_mc_local)
        acc_sum = acc_sum.at[action_id, chain_idx].add(accepted)
        sel_sum = sel_sum.at[action_id, chain_idx].add(1)
        return [state, key, log_amps, acc_sum, sel_sum]

    state, _, log_amps, acc_sum, sel_sum = jax.lax.fori_loop(
        0, nsweeps * Ns, _step, [state, key, log_amps, acc_sum, sel_sum]
    )

    acc_total  = jax.lax.psum(jnp.sum(acc_sum, axis=1), 'i')
    sel_total  = jax.lax.psum(jnp.sum(sel_sum, axis=1), 'i')
    acceptance = acc_total / sel_total
    return state, log_amps, acceptance
