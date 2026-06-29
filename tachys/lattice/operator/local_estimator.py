from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.experimental.shard_map import shard_map

from tachys.lattice.operator.base import DiagonalResult, OffdiagonalResult, DiagOffdiagResult
from tachys.parallel import mesh, n_devices


def _boolean_partition_indices(mask: jnp.ndarray):
    """Return (perm, perm_inv) that partitions mask True-first.

    perm     — permutation such that mask[perm] has all True entries first
    perm_inv — inverse permutation (perm_inv[perm[i]] == i)
    """
    n = mask.size
    n_true = mask.sum()
    true_positions  = jnp.cumsum(mask) - 1
    false_positions = jnp.cumsum(~mask) - 1 + n_true
    dest     = jnp.where(mask, true_positions, false_positions)
    perm     = jnp.empty_like(dest).at[dest].set(jnp.arange(n))
    perm_inv = dest
    return perm, perm_inv


def _apply_masked(wf, connected_states, mask, log_amp, batch_expand=1):
    """Compute log-amplitudes for connected states, skipping all-zero-mask batches.

    Flattens (N_terms, N_mc) into a single sequence, sorts active connections
    to the front, then rebatches at size ``batch_expand * N_mc``.  A
    ``while_loop`` then stops as soon as it reaches the first all-zero batch,
    avoiding network evaluations for guaranteed-zero contributions.

    Parameters
    ----------
    connected_states : pytree, leaves shape (N_terms, N_mc, ...)
    mask             : (N_terms, N_mc) — nonzero entries mark active connections
    log_amp          : (N_mc,) — current log-amplitudes, used only for dtype
    batch_expand     : int — batch enlargement factor; N_terms must be divisible

    Returns
    -------
    log_amps_connected : (N_terms, N_mc)
    """
    active = mask.astype(bool)
    N_terms, N_mc = active.shape
    assert (N_terms * N_mc) % batch_expand == 0

    # Flatten (N_terms, N_mc) -> (N_terms * N_mc,)
    flat_active = active.reshape(-1)
    flat_states = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), connected_states)

    # Sort: active connections first
    perm, perm_inv = _boolean_partition_indices(flat_active)
    flat_active = flat_active[perm]
    flat_states = jax.tree.map(lambda x: x[perm], flat_states)

    # Rebatch: new batch size = batch_expand * N_mc
    batch_size = batch_expand * N_mc
    n_batches  = (N_terms * N_mc) // batch_size
    flat_states = jax.tree.map(
        lambda x: x.reshape(n_batches, batch_size, *x.shape[1:]), flat_states
    )
    flat_active = flat_active.reshape(n_batches, batch_size)

    batch_has_active = flat_active.sum(1) > 0
    log_amps_connected = jnp.zeros(flat_active.shape, dtype=log_amp.dtype)

    def body_fun(val):
        i, log_amps_connected = val
        batch_states   = jax.tree.map(lambda x: x[i], flat_states)
        log_amps_batch = wf.apply_fn(wf.params, batch_states)
        return i + 1, log_amps_connected.at[i].set(log_amps_batch)

    def cond_fun(val):
        i, _ = val
        return batch_has_active[i] & (i < batch_has_active.size)

    _, log_amps_connected = jax.lax.while_loop(cond_fun, body_fun, (0, log_amps_connected))

    # Undo rebatching and reordering -> (N_terms, N_mc)
    return log_amps_connected.reshape(-1)[perm_inv].reshape(N_terms, N_mc)


def local_estimator(operator, state, wf, log_amp, optimize_mask=True, batch_expand=1):
    """Local estimator O_L(x) = sum_{x'} <x|O|x'> * psi(x') / psi(x).

    Parameters
    ----------
    operator      : _Operator or _OperatorSum
    state         : State, batch axis 0 of size N_mc
    wf            : wave function with .apply_fn(params, state) -> (N_mc,) log-amplitudes
    log_amp       : jax.Array, shape (N_mc,)
    optimize_mask : bool — skip zero-mask batches via while_loop (default True)
    batch_expand  : int — batch enlargement factor for _apply_masked (N_terms must
                    be divisible; only used when optimize_mask=True)

    Returns
    -------
    O_L : jax.Array, shape (N_mc,)
    """
    result = operator(state)

    def _diagonal(diag):
        return jnp.sum(diag.matrix_element, axis=0)

    def _offdiagonal(offdiag):
        if optimize_mask:
            log_amps_connected = _apply_masked(
                wf, offdiag.connected_states, offdiag.mask, log_amp,
                batch_expand=batch_expand,
            )
        else:
            log_amps_connected = jax.lax.map(
                lambda s: wf.apply_fn(wf.params, s),
                offdiag.connected_states,
            )
        psi_ratio = jnp.exp(log_amps_connected - log_amp[None, :])
        return jnp.sum(offdiag.matrix_element * offdiag.mask * psi_ratio, axis=0)

    if isinstance(result, DiagOffdiagResult):
        return _diagonal(result.diagonal) + _offdiagonal(result.offdiagonal)
    if isinstance(result, DiagonalResult):
        return _diagonal(result)
    if isinstance(result, OffdiagonalResult):
        return _offdiagonal(result)
    raise TypeError(f"Unexpected operator result type: {type(result)}")


@partial(jax.jit, static_argnames=('optimize_mask', 'batch_expand'))
def compute_expectation(operator, wf, state, log_amp, optimize_mask=True, batch_expand=1):
    """Sharded expectation value of an operator.

    Shards state and log_amp across all devices, evaluates local_estimator on
    each shard, then reduces to global statistics via psum.

    Parameters
    ----------
    operator      : _Operator or _OperatorSum — replicated across devices
    wf            : wave function — replicated across devices
    state         : State, batch axis 0 sharded across devices
    log_amp       : jax.Array, shape (N_mc,) sharded across devices
    optimize_mask : bool
    batch_expand  : int

    Returns
    -------
    O_L     : jax.Array, shape (N_mc,) — local estimator, sharded
    O_mean  : scalar — global mean <O>
    O2_mean : scalar — global mean <|O|^2>
    """
    def _body(operator, wf, state, log_amp):
        O_L     = local_estimator(operator, state, wf, log_amp,
                                  optimize_mask=optimize_mask,
                                  batch_expand=batch_expand)
        O_mean  = jax.lax.psum(jnp.mean(O_L),               'i') / n_devices
        O2_mean = jax.lax.psum(jnp.mean(jnp.abs(O_L) ** 2), 'i') / n_devices
        return O_L, O_mean, O2_mean

    return shard_map(
        _body,
        mesh=mesh,
        in_specs=(P(None), P(None), P('i'), P('i')),
        out_specs=(P('i'), P(), P()),
        check_rep=False,
    )(operator, wf, state, log_amp)
