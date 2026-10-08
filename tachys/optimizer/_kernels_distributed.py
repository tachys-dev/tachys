"""Kernel backend that keeps the NTK row-sharded and solves it with JAXMg.

Selected with ``kernels="distributed"`` on SR, SPRING or MARCH. The
default backend, ``_kernels``, psums the whole NTK onto every device and solves
it there. Here each device builds and keeps only its own rows of the kernel, and
JAXMg's ``potrs_shardmap_ctx`` (cuSOLVERMp, one process per GPU) solves the
row-sharded system across all of them. Both backends are called from the same
place, inside the optimizer step's shard_map over 'i'. They take the same
arguments and return the solution in the same layout, so the rest of the step
(centring, VJP, momentum) is shared.

Only the Cholesky solver of SR, SPRING and MARCH is distributed so far. TDVP,
which solves with eigh, takes ``kernels="default"`` only.

Row order
---------
In mode="real" device ``d`` holds rows ``d*m ... (d+1)*m - 1`` of the
``M x M`` kernel, with ``m = M / n_devices`` chains per device.

In mode="complex" the dense ``2M x 2M`` kernel is in *device-major* order, so
that every device holds the real and the imaginary rows of its own chains.
Row (and column) ``d*2m + a*m + i`` is component ``a`` of chain ``d*m + i``.
This differs from the *component-major* order ``a*M + j`` of
``_kernels._ntk_to_dense``. The solver therefore permutes the right-hand side
into device-major order (``_to_solver_order``) and the solution back
(``_from_solver_order``), and returns the component-major ``[u, v]`` that
``center_sr_solution`` expects.
"""

import os

import jax
import jax.numpy as jnp
from einops import rearrange
from jax.sharding import PartitionSpec as P

from tachys.lattice.foundation.collectives import grouped_mean
from tachys.lattice.foundation.foundation_state import FoundationState
from tachys.lattice.state_array import get_n_mc_local
from tachys.optimizer._kernels import (
    _matmul_precision,
    _ntk_contraction,
    _ntk_setup,
    _solver_rhs,
)
from tachys.parallel import hard_shard, n_devices

# The sharding of the dense kernel compute_ntk returns: rows over 'i'.
NTK_SPEC = P('i', None)

# Tile size of cuSOLVERMp's block-cyclic layout (see _tile_size).
_DEFAULT_MAX_TILE = 1024
_MIN_TILE = 128


# ─── NTK rows ─────────────────────────────────────────────────────────────────

def ntk_rows_fn(state, wf, nbatches, mode, V=None, dtype=None):
    """This device's rows of the uncentred NTK; call inside shard_map over 'i'.

    Returns (m, M) [real] or (m, M, 2, 2) [complex]: row i is local chain i, and
    column j is global chain j. Each of the device's own batches is contracted
    with every batch of every device. Unlike ``_kernels.ntk_parallel_fn``, no
    device ever holds, or psums, the whole M x M kernel. The cost is the same
    order: ``nbatches + n_devices * nbatches**2`` Jacobian evaluations per
    device.
    """
    global_state, jacobian, V = _ntk_setup(state, wf, nbatches, mode, V=V, dtype=dtype)

    N_batches      = get_n_mc_local(global_state)
    N_mc_per_batch = get_n_mc_local(state) // nbatches
    first_batch    = jax.lax.axis_index('i') * nbatches  # this device's first batch

    shape = (nbatches, N_mc_per_batch, N_batches, N_mc_per_batch)
    rows = jnp.zeros(shape + ((2, 2) if mode == "complex" else ()))

    def batch_rows(i, rows):
        state_i = jax.tree.map(lambda x: x[first_batch + i], global_state)
        with _matmul_precision(dtype):
            J1 = jacobian(state_i)

        def block(j, rows):
            state_j = jax.tree.map(lambda x: x[j], global_state)
            with _matmul_precision(dtype):
                J2     = jacobian(state_j)
                ntk_ij = _ntk_contraction(J1, J2, mode, V=V)
            return rows.at[i, :, j].set(ntk_ij)

        return jax.lax.fori_loop(0, N_batches, block, rows)

    rows = jax.lax.fori_loop(0, nbatches, batch_rows, rows)
    return rearrange(rows, 'i a j b ... -> (i a) (j b) ...')


def _local_grouped_mean(x, y, K, axis):
    """Each element of x replaced by the mean of its group along ``axis``.

    The same as ``grouped_mean(x, y, K, axis, broadcast=True)``, but with no
    psum over 'i'. Use it for an axis that every device holds whole, such as
    the columns of the row-sharded kernel. There, grouped_mean's psum would add
    together the rows of different devices.
    """
    total  = jax.ops.segment_sum(jnp.moveaxis(x, axis, 0), y, num_segments=K)
    counts = jax.ops.segment_sum(jnp.ones_like(y, dtype=x.dtype), y, num_segments=K)
    mean   = total / jnp.reshape(counts, (K,) + (1,) * (x.ndim - 1))
    return jnp.moveaxis(jnp.take(mean, y, axis=0), 0, axis)


def center_ntk_rows(rows, weights, state):
    """``_kernels.center_ntk`` on this device's rows of the kernel.

    The means over rows (axis 0) combine every device's rows with a psum. The
    means over columns (axis 1) are local, because every device holds all the
    columns of its rows. The result equals the rows of the centred replicated
    kernel, up to the order of the sums.
    """
    M = rows.shape[1]
    if isinstance(state, FoundationState):
        if weights is not None:
            raise NotImplementedError(
                "center_ntk_rows: weights + FoundationState is not supported yet."
            )
        ids_all     = jax.lax.all_gather(state.system_ids, 'i', tiled=True)
        K           = state.n_systems
        row_mean    = grouped_mean(rows, state.system_ids, K, axis=0, broadcast=True)
        col_mean    = _local_grouped_mean(rows, ids_all, K, axis=1)
        global_mean = _local_grouped_mean(row_mean, ids_all, K, axis=1)
    elif weights is None:
        row_mean    = jax.lax.psum(jnp.sum(rows, axis=0, keepdims=True), 'i') / M
        col_mean    = jnp.mean(rows, axis=1, keepdims=True)
        global_mean = jnp.mean(row_mean, axis=1, keepdims=True)
    else:
        trailing    = tuple(range(1, rows.ndim - 1))
        w_rows      = jnp.expand_dims(weights, axis=trailing)
        w_cols      = jnp.expand_dims(jax.lax.all_gather(weights, 'i', tiled=True), axis=trailing)
        row_mean    = jax.lax.psum(jnp.sum(w_rows[:, None] * rows, axis=0, keepdims=True), 'i') / M
        col_mean    = jnp.mean(w_cols[None, :] * rows, axis=1, keepdims=True)
        global_mean = jax.lax.psum(
            jnp.sum(w_rows[:, None] * rows * w_cols[None, :], axis=(0, 1), keepdims=True), 'i'
        ) / M ** 2

    return rows - row_mean - col_mean + global_mean


def compute_ntk(state, wf, mode, weights=None, V=None, nbatches=1, dtype=None):
    """This device's rows of the centred, weighted kernel, made dense.

    The row-sharded counterpart of ``_kernels.compute_ntk``, with the same
    arguments; call inside shard_map over 'i'. Returns the (m, M) rows [real],
    or the (2m, 2M) rows of the device-major dense kernel [complex] (see the
    module docstring). Shard it with ``NTK_SPEC``.
    """
    rows = ntk_rows_fn(state, wf, nbatches, mode, V=V, dtype=dtype)
    rows = center_ntk_rows(rows, weights, state)

    if weights is not None:
        sqrt_w_rows = jnp.sqrt(weights)
        sqrt_w_cols = jax.lax.all_gather(jnp.sqrt(weights), 'i', tiled=True)
        if mode == "complex":
            rows = rows * sqrt_w_rows[:, None, None, None] * sqrt_w_cols[None, :, None, None]
        else:
            rows = rows * sqrt_w_rows[:, None] * sqrt_w_cols[None, :]

    if mode == "complex":
        rows = rearrange(rows, 'i (e j) a b -> (a i) (e b j)', e=n_devices)
    return rows


# ─── Row order ────────────────────────────────────────────────────────────────

def _to_solver_order(v, mode):
    """A component-major vector ``[real part; imaginary part]`` of length 2M,
    in the device-major order of the kernel's rows (identity in mode="real")."""
    if mode == "real":
        return v
    return rearrange(v, '... (a d i) -> ... (d a i)', a=2, d=n_devices)


def _from_solver_order(v, mode):
    """Inverse of ``_to_solver_order``."""
    if mode == "real":
        return v
    return rearrange(v, '... (d a i) -> ... (a d i)', a=2, d=n_devices)


# ─── Cholesky solve ───────────────────────────────────────────────────────────

def _add_diag_shift(rows, shift):
    """Add ``shift`` to the global diagonal of the row-sharded kernel.

    Call inside shard_map over 'i'. Local row k is global row
    ``axis_index('i') * n_local + k``.
    """
    n_local = rows.shape[0]
    k = jnp.arange(n_local)
    start = jax.lax.axis_index('i') * n_local
    return rows.at[k, start + k].add(jnp.asarray(shift, dtype=rows.dtype))


def _tile_size(n_local):
    """The tile size T_A of cuSOLVERMp's 2D block-cyclic layout, for local
    blocks of ``n_local`` rows.

    JAXMg pads every local block to a multiple of T_A, which costs a copy of
    the matrix. The largest power of two that divides n_local avoids that copy,
    if it is at least 128 or the whole block. Otherwise there is no good
    divisor, and the largest power of two up to n_local keeps the padding
    small. The cap, 1024 by default, can be lowered with the environment
    variable TACHYS_JAXMG_TILE_SIZE. It bounds the redistribution scratch,
    which is ``3 * T_A * N`` elements per process.
    """
    cap = int(os.environ.get("TACHYS_JAXMG_TILE_SIZE", _DEFAULT_MAX_TILE))
    if cap < 1:
        raise ValueError(f"TACHYS_JAXMG_TILE_SIZE must be a positive integer, got {cap}")
    cap = 1 << (cap.bit_length() - 1)               # a power of two

    divisor = min(n_local & -n_local, cap)          # largest power of two dividing n_local
    if divisor >= _MIN_TILE or divisor == n_local:
        return divisor
    return min(cap, 1 << (n_local.bit_length() - 1))


def _jaxmg():
    """The jaxmg module, imported on first use: CPU installs don't have it."""
    try:
        import jaxmg
    except ImportError as err:
        raise ImportError(
            "kernels='distributed' solves with JAXMg, which is not installed: "
            'pip install "tachys[jaxmg]"'
        ) from err
    # potrs_jit_ctx came with the potrs_shardmap_ctx that takes local blocks;
    # before it, potrs_shardmap_ctx took global arrays.
    if not hasattr(jaxmg, "potrs_jit_ctx"):
        raise ImportError(
            f"JAXMg {jaxmg.__version__} has no potrs_shardmap_ctx that solves on "
            "local blocks inside shard_map, which kernels='distributed' needs."
        )
    return jaxmg


def _potrs(a_rows, b_rows):
    """Solve the row-sharded SPD system with JAXMg; call inside shard_map over 'i'.

    a_rows : this device's rows of the matrix (only the lower triangle is read).
    b_rows : this device's rows of the right-hand side, a vector: sharded like
             the matrix (``NTK_SPEC``), which is what potrs_shardmap_ctx
             expects of its right-hand side.

    Returns this device's rows of the solution, all NaN on every device if the
    factorization failed on any of them. JAXMg reports a failure, such as a
    matrix that is not positive definite, in the status vector and may leave a
    finite but meaningless solution.
    """
    _, x, status = _jaxmg().potrs_shardmap_ctx(
        a_rows, b_rows, _tile_size(a_rows.shape[0]), NTK_SPEC,
    )
    failed = jax.lax.pmax((status[0] != 0).astype(jnp.int32), 'i') > 0  # status[0]: status_code
    return jnp.where(failed, jnp.nan, x)


def linear_solver_cholesky(ntk, eps, diag_shift, mode="complex"):
    """Cholesky solve of the SR system on the row-sharded kernel.

    Drop-in replacement for ``_kernels.linear_solver_cholesky``, inside
    shard_map over 'i'. Same arguments and return value, except that ``ntk``
    is this device's rows from ``compute_ntk``. ``eps`` is the replicated force
    vector of shape (M,). Returns the replicated solution: (M,) [real], or
    (2M,) ``[u, v]`` with x = u + iv [complex].

    In mode="complex" the 2M x 2M system is solved in one piece, rather than
    through the Schur complement of its real block as the default solver does.
    A failed factorization gives a zero update, as in the default solver.
    """
    if mode not in ("real", "complex"):
        raise ValueError("mode must be 'real' or 'complex'")

    A = _add_diag_shift(ntk, diag_shift)
    b = hard_shard(_to_solver_order(_solver_rhs(eps, mode), mode))
    sol = jax.lax.all_gather(_potrs(A, b), 'i', tiled=True)
    sol = _from_solver_order(sol, mode)
    return jnp.where(jnp.all(jnp.isfinite(sol), axis=-1, keepdims=True), sol, 0.0)
