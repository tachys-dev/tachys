from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax import shard_map

from tachys.lattice.operator.base import DiagonalResult, OffdiagonalResult, DiagOffdiagResult
from tachys.parallel import mesh, n_devices


def _boolean_partition_indices(mask: jnp.ndarray):
    """Return (perm, perm_inv) that partitions mask True-first.

    perm     — permutation such that mask[perm] has all True entries first
    perm_inv — inverse permutation (perm_inv[perm[i]] == i)
    """
    n = mask.size
    # Stable argsort on a boolean key is a counting sort (O(N)); ~mask puts
    # True entries (0) before False entries (1) in ascending order.
    perm     = jnp.argsort(~mask, stable=True)
    perm_inv = jnp.empty(n, dtype=perm.dtype).at[perm].set(jnp.arange(n, dtype=perm.dtype))
    return perm, perm_inv


def _apply_masked(wf, connected_states, mask, log_amp, batch_expand=1):
    """Compute log-amplitudes for connected states, skipping all-zero-mask batches.

    Flattens (N_terms, N_mc_local) into a single sequence, sorts active connections
    to the front, then rebatches at size ``batch_expand * N_mc_local``.  A
    ``while_loop`` then stops as soon as it reaches the first all-zero batch,
    avoiding network evaluations for guaranteed-zero contributions.

    Parameters
    ----------
    connected_states : pytree, leaves shape (N_terms, N_mc_local, ...)
    mask             : (N_terms, N_mc_local) — nonzero entries mark active connections
    log_amp          : (N_mc_local,) — current log-amplitudes, used only for dtype
    batch_expand     : float — batch scale factor; int(batch_expand * N_mc_local) must divide N_terms * N_mc_local

    Returns
    -------
    log_amps_connected : (N_terms, N_mc_local)
    """
    active = mask.astype(bool)
    N_terms, N_mc_local = active.shape
    batch_size = int(batch_expand * N_mc_local)
    assert batch_size > 0 and (N_terms * N_mc_local) % batch_size == 0

    # Flatten (N_terms, N_mc_local) -> (N_terms * N_mc_local,)
    flat_active = active.reshape(-1)
    flat_states = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), connected_states)

    # Sort: active connections first
    perm, perm_inv = _boolean_partition_indices(flat_active)
    flat_active = flat_active[perm]
    flat_states = jax.tree.map(lambda x: x[perm], flat_states)

    # Rebatch: new batch size = int(batch_expand * N_mc_local)
    n_batches  = (N_terms * N_mc_local) // batch_size
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

    # Undo rebatching and reordering -> (N_terms, N_mc_local)
    return log_amps_connected.reshape(-1)[perm_inv].reshape(N_terms, N_mc_local)


def _offdiagonal_terms(offdiag, wf, log_amp, optimize_mask=True, batch_expand=1):
    """Per-term, unsummed off-diagonal contributions to <x|O|x'> * psi(x') / psi(x).

    Parameters
    ----------
    offdiag       : OffdiagonalResult
    wf            : wave function with .apply_fn(params, state) -> (N_mc_local,) log-amplitudes
    log_amp       : jax.Array, shape (N_mc_local,)
    optimize_mask : bool — skip zero-mask batches via while_loop (default True)
    batch_expand  : float — batch scale factor for _apply_masked (only used when optimize_mask=True)

    Returns
    -------
    matrix_element, mask, psi_ratio : each jax.Array, shape (N_terms, N_mc_local)
        The summand is ``matrix_element * mask * psi_ratio``; callers that need the
        per-term breakdown (e.g. DMC's Green's-function step) use these directly,
        while ``local_estimator`` sums over the term axis.
    """
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
    return offdiag.matrix_element, offdiag.mask, psi_ratio


def local_estimator(operator, state, wf, log_amp, optimize_mask=True, batch_expand=1):
    """Local estimator O_L(x) = sum_{x'} <x|O|x'> * psi(x') / psi(x).

    Parameters
    ----------
    operator      : _Operator or _OperatorSum
    state         : State, batch axis 0 of size N_mc_local
    wf            : wave function with .apply_fn(params, state) -> (N_mc_local,) log-amplitudes
    log_amp       : jax.Array, shape (N_mc_local,)
    optimize_mask : bool — skip zero-mask batches via while_loop (default True)
    batch_expand  : float — batch scale factor for _apply_masked (only used when optimize_mask=True)

    Returns
    -------
    O_L : jax.Array, shape (N_mc_local,)
    """
    result = operator(state)

    def _diagonal(diag):
        return jnp.sum(diag.matrix_element, axis=0)

    def _offdiagonal(offdiag):
        matrix_element, mask, psi_ratio = _offdiagonal_terms(
            offdiag, wf, log_amp, optimize_mask=optimize_mask, batch_expand=batch_expand,
        )
        return jnp.sum(matrix_element * mask * psi_ratio, axis=0)

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
    log_amp       : jax.Array, shape (N_mc_local,) sharded across devices
    optimize_mask : bool
    batch_expand  : int

    Returns
    -------
    O_L     : jax.Array, shape (N_mc_local,) — local estimator, sharded
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
        in_specs=(P(),     P(),     P('i'), P('i')),
        out_specs=(P('i'), P(), P()),
        check_vma=False,
    )(operator, wf, state, log_amp)
