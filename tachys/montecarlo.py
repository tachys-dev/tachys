from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from jax.sharding import PartitionSpec as P
from jax.experimental.shard_map import shard_map

from tachys.parallel import mesh, n_devices
from tachys.lattice.operator.local_estimator import _apply_masked


class _BaseAction(struct.PyTreeNode):
    """Base class for MCMC move proposals.

    Subclasses must implement ``__call__(self, key, state)`` returning
    ``(new_state, allowed_move, log_prob_correction)`` where:

    - ``new_state``: proposed State after applying the move.
    - ``allowed_move``: boolean mask (N_mc,) — False when the move is a
      no-op (e.g. exchanging identical spins or identical occupations),
      allowing early rejection before evaluating the wavefunction.
    - ``log_prob_correction``: scalar log-probability correction for
      asymmetric proposals; 0.0 for symmetric moves.

    ``state`` can be any subclass of ``State``.
    """


def _cast_floating_to(tree, dtype):
    """Cast all floating-point leaves of a pytree to ``dtype``."""
    def conditional_cast(x):
        if isinstance(x, (np.ndarray, jnp.ndarray)) and jnp.issubdtype(x.dtype, jnp.floating):
            x = x.astype(dtype)
        return x
    return jax.tree_util.tree_map(conditional_cast, tree)


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
    state, key, log_amps, accepted
        ``accepted`` is a boolean array of shape (N_mc,).
    """
    keys = jax.vmap(partial(jax.random.split, num=3))(key)
    key, subkey1, subkey2 = keys.T

    new_state, allowed_move, log_prob_correction = action(subkey1, state)

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

    return state, key, log_amps, accepted


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
        ``acceptance`` is the global mean acceptance rate over all chains and steps.
    """
    wf, state = _cast_floating_to((wf, state), wf.dtype)

    log_amps = wf.apply_fn(wf.params, state)
    Ns       = state.Ns
    N_mc_local = state.spins.shape[0]
    acc_sum    = jnp.zeros(N_mc_local)

    def _step(_, vals):
        state, key, log_amps, acc_sum = vals
        state, key, log_amps, accepted = mc_step(state, key, action, wf, log_amps)
        return [state, key, log_amps, acc_sum + accepted]

    state, _, log_amps, acc_sum = jax.lax.fori_loop(
        0, nsweeps * Ns, _step, [state, key, log_amps, acc_sum]
    )

    acceptance = jax.lax.psum(jnp.mean(acc_sum), 'i') / (n_devices * nsweeps * Ns)
    return state, log_amps, acceptance
