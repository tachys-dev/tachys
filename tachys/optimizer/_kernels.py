"""Low-level kernels for SR-family optimizers.

Ported from the jaxvmc reference implementation; adapted for tachys naming
conventions (state / wf instead of lattice / vstate).
"""

import operator
from itertools import combinations_with_replacement

from einops import rearrange, einsum
import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular

from tachys.parallel import n_devices, rank
from tachys.lattice.foundation.foundation_state import FoundationState
from tachys.lattice.foundation.collectives import grouped_mean
from tachys.lattice.state_array import get_n_mc, get_n_mc_local

# ─── Linear solver ────────────────────────────────────────────────────────────

def linear_solver_cholesky(ntk, eps, diag_shift, mode="complex"):
    """Cholesky-based solver for the SR linear system.

    mode="real"   : ntk is (..., M, M) real SPD; returns (..., M).
    mode="complex": ntk is (..., M, M, 2, 2) block matrix; eps is (..., M)
                    complex; returns (..., 2*M) real [u, v] with x = u + iv.
    """
    if mode == "real":
        A = ntk
        n = A.shape[-1]
        idx = jnp.arange(n)
        A = A.at[..., idx, idx].add(jnp.asarray(diag_shift, dtype=A.dtype))
        L = jnp.linalg.cholesky(A)
        y = solve_triangular(L, eps.real, lower=True, trans=0)
        u = solve_triangular(L, y, lower=True, trans=1)
        # jnp.linalg.cholesky doesn't raise on a non-PD input under jit; it
        # fills the offending row (and every row after it) with NaN, which
        # would otherwise silently poison every parameter for the rest of
        # training. Degrade to a no-op update instead.
        return jnp.where(jnp.all(jnp.isfinite(u), axis=-1, keepdims=True), u, 0.0)

    elif mode == "complex":
        eR, eI = eps.real, eps.imag
        A = ntk[..., 0, 0]
        B = ntk[..., 0, 1]
        C = ntk[..., 1, 1]
        n = A.shape[-1]
        idx = jnp.arange(n)
        shift = jnp.asarray(diag_shift, dtype=A.dtype)
        A = A.at[..., idx, idx].add(shift)
        C = C.at[..., idx, idx].add(shift)

        L = jnp.linalg.cholesky(A)
        W   = solve_triangular(L, B,  lower=True, trans=0)
        w_e = solve_triangular(L, eR, lower=True, trans=0)

        WT = jnp.swapaxes(W, -1, -2)
        S  = C - WT @ W
        S  = 0.5 * (S + jnp.swapaxes(S, -1, -2))
        rhs_v = -(eI + WT @ w_e)

        Ls = jnp.linalg.cholesky(S)
        yv = solve_triangular(Ls, rhs_v, lower=True, trans=0)
        v  = solve_triangular(Ls, yv,   lower=True, trans=1)

        y = w_e - W @ v
        u = solve_triangular(L, y, lower=True, trans=1)
        sol = jnp.concatenate([u, v], axis=-1)
        # See the "real" branch above: guard against a non-PD pivot (in
        # either cholesky call) silently NaN-poisoning the whole update.
        return jnp.where(jnp.all(jnp.isfinite(sol), axis=-1, keepdims=True), sol, 0.0)

    else:
        raise ValueError("mode must be 'real' or 'complex'")



def _ntk_to_dense(ntk, mode):
    """Flatten the NTK into the dense real symmetric matrix the solvers invert.

    mode="real"    : (..., M, M) — returned unchanged.
    mode="complex" : (..., M, M, 2, 2) -> (..., 2M, 2M) in the block form
                     ``[[A, B], [B.T, C]]`` with ``A = ntk[..., 0, 0]``,
                     ``B = ntk[..., 0, 1]``, ``C = ntk[..., 1, 1]`` — exactly the
                     system ``linear_solver_cholesky`` assembles by hand.

    The row index of the dense matrix is ``a * M + i`` (component-major: the whole
    real block first, then the whole imaginary one), *not* the interleaved
    ``i * 2 + a``. That ordering is load-bearing: the solvers return
    ``concatenate([u, v])`` and ``center_sr_solution`` un-stacks it with
    ``reshape(2, -1).T``, which only reproduces ``(u, v)`` columns for the
    component-major layout. The interleaved one silently yields a finite but
    wrong solution.
    """
    if mode == "real":
        return ntk
    M = ntk.shape[-3]
    # axes (..., i, j, a, b) -> (..., a, i, b, j), then C-order reshape gives
    # element [a*M + i, b*M + j] = ntk[i, j, a, b].
    G = jnp.moveaxis(ntk, (-2, -1), (-4, -2))
    return G.reshape(*ntk.shape[:-4], 2 * M, 2 * M)


def _solver_rhs(eps, mode):
    """Right-hand side vector in the same row order as ``_ntk_to_dense``.

    Note the minus sign on the imaginary part: the block system solved in
    complex mode is ``[[A, B], [B.T, C]] [u; v] = [Re eps; -Im eps]`` (see the
    Schur-complement algebra in ``linear_solver_cholesky``), so a plain
    ``concatenate([eps.real, eps.imag])`` looks right and is not.
    """
    if mode == "real":
        return eps.real
    return jnp.concatenate([eps.real, -eps.imag], axis=-1)


def linear_solver_eigh(ntk, eps, diag_shift, mode="complex", rcond=1e-8, atol=0.0):
    """Spectrally-truncated (pseudo-inverse) solver for the SR/TDVP linear system.

    Diagonalizes the kernel and inverts it only on the eigenvectors whose
    eigenvalue clears ``cutoff = max(rcond * lambda_max, atol)``, projecting the
    rest away. This is the regularization of choice for real-time dynamics, where
    the kernel is genuinely rank deficient — ``center_ntk`` alone puts exact zero
    modes in the spectrum, and ``2M > P`` makes it singular by construction — and
    where Tikhonov damping distorts the well-resolved directions instead of
    removing the unresolved ones.

    Drop-in replacement for ``linear_solver_cholesky``: same arguments, same
    return layout, so ``center_sr_solution`` and the VJP downstream are unchanged.

    Parameters
    ----------
    ntk        : jax.Array — (..., M, M) real SPD in mode="real";
                 (..., M, M, 2, 2) block matrix in mode="complex".
    eps        : jax.Array — force vector, shape (..., M); complex in mode="complex".
    diag_shift : float — Tikhonov shift applied to the *kept* eigenvalues,
                 ``1 / (lambda + diag_shift)``. Because ``diag_shift * I`` is
                 isotropic it commutes with the eigendecomposition, so there is no
                 need to modify the matrix before ``eigh``. Pass 0.0 to rely on the
                 truncation alone (the recommended setting for dynamics).
    mode       : "real" or "complex".
    rcond      : relative eigenvalue cutoff — modes with
                 ``lambda <= rcond * lambda_max`` are discarded. Relative rather than
                 absolute because the kernel's scale varies by orders of magnitude
                 with ansatz, system size and step (``_build_ntk`` divides by the
                 chain count, MARCH rescales the Jacobian), whereas ``1 / rcond`` is
                 exactly the condition number the retained subspace is capped at.
    atol       : absolute floor on the cutoff. Default 0.0 (purely relative).

    Returns
    -------
    (..., M) real in mode="real", or (..., 2M) real ``[u, v]`` (solution
    ``x = u + iv``) in mode="complex" — matching ``linear_solver_cholesky``.

    Notes
    -----
    The keep mask is evaluated on the *raw* spectrum, before ``diag_shift`` is
    applied, so the two regularizers stay orthogonal: ``rcond`` chooses the
    retained subspace, ``diag_shift`` softens the amplification inside it. Masking
    ``lambda + diag_shift`` instead would let a large enough shift silently switch
    the truncation off.

    The mask is ``lambda > cutoff``, not ``|lambda| > cutoff``: the kernel is a Gram
    matrix, so a negative eigenvalue is pure roundoff on a direction carrying no
    signal, and admitting it would flip the update along that direction and
    amplify it by ``1 / |lambda|``.

    A hard truncation makes the solution discontinuous in the parameters whenever
    an eigenvalue crosses the threshold, which is harmless for a fixed-step
    integrator but would corrupt an embedded error estimate used for step-size
    control.
    """
    if mode not in ("real", "complex"):
        raise ValueError("mode must be 'real' or 'complex'")

    G = _ntk_to_dense(ntk, mode)
    b = _solver_rhs(eps, mode)

    # eigh reads a single triangle; symmetrizing makes an asymmetric input an
    # averaging error rather than a silent choice of triangle.
    G = 0.5 * (G + jnp.swapaxes(G, -1, -2))
    lam, U = jnp.linalg.eigh(G)                         # ascending, real

    lam_max = jnp.maximum(jnp.max(lam, axis=-1, keepdims=True), 0.0)
    # Never keep a mode below the float resolution of the largest eigenvalue, even
    # when both rcond and atol are 0 — that also makes the all-junk spectrum case
    # (lam_max <= 0) return an exact zero update instead of exploding.
    cutoff = jnp.maximum(jnp.maximum(rcond * lam_max, atol),
                         jnp.finfo(lam.dtype).eps * lam_max)

    keep = lam > cutoff
    # Divide by a safe value, never by the discarded (possibly zero or negative)
    # eigenvalue: 1/0 is inf and inf * 0 = NaN survives the jnp.where.
    lam_safe = jnp.where(keep, lam, 1.0)
    inv = jnp.where(keep, 1.0 / (lam_safe + jnp.asarray(diag_shift)), 0.0)

    sol = jnp.einsum('...ij,...j->...i', U, inv * jnp.einsum('...ji,...j->...i', U, b))

    # Same guard as linear_solver_cholesky: a non-finite kernel makes eigh return
    # all-NaN, and the keep mask does not stop 0 * NaN from propagating through U.
    # Degrade to a no-op update rather than poisoning every parameter.
    return jnp.where(jnp.all(jnp.isfinite(sol), axis=-1, keepdims=True), sol, 0.0)


# ─── NTK computation ──────────────────────────────────────────────────────────

def _ntk_contraction(J1, J2, mode, V=None):
    """Contract pytree Jacobians into the NTK block for a pair of batches.

    J1, J2 : pytrees with leaves of shape (N_i, *leaf_shape)       [real]
                                           (N_i, 2, *leaf_shape)    [complex]
    V      : pytree matching wf.params (MARCH preconditioner), optional.
             Each leaf has shape (*leaf_shape); broadcast is automatic.
    Returns: (N_i, N_j)           [real]
             (N_i, N_j, 2, 2)    [complex]
    """
    def contract(x, y, v=None):
        if v is not None:
            x = x / (v ** 0.5 + 1e-8)
        if mode == "complex":
            return einsum(x, y, 'i j ..., k l ... -> i k j l')
        else:
            return einsum(x, y, 'i ..., j ... -> i j')

    if V is not None:
        pairs = jax.tree.map(contract, J1, J2, V)
    else:
        pairs = jax.tree.map(lambda x, y: contract(x, y), J1, J2)

    return jax.tree.reduce(operator.add, pairs)


def ntk_parallel_fn(state, wf, nbatches, mode, V=None):
    """Assemble the full NTK matrix using distributed pair-wise Jacobian contractions.

    Each device computes a subset of (batch_i, batch_j) pairs; contributions
    are summed via psum to yield the (N_mc × N_mc) NTK.
    """
    global_state = jax.lax.all_gather(state, 'i')  # array: (n_devices, N_mc_local, N)

    if nbatches > 1:
        global_state = jax.tree.map(
            lambda x: rearrange(
                x, 'P (nbatches Nmc) ... -> (P nbatches) Nmc ...', nbatches=nbatches
            ),
            global_state,
        )

    N_batches      = get_n_mc_local(global_state)
    N_mc_per_batch = get_n_mc_local(state) // nbatches

    # Build jacobian_fn once outside body_fun so it is compiled once.
    if mode == "complex":
        def _f(params, s):
            log_amps = jnp.squeeze(wf.apply_fn(params, s))
            return jnp.stack([log_amps.real, log_amps.imag])  # (2,)
    else:
        _f = lambda params, s: jnp.squeeze(wf.apply_fn(params, s)).real  # scalar

    # vmap over samples: leaves become (N_mc, 2, *leaf_shape) or (N_mc, *leaf_shape).
    jacobian_fn = jax.vmap(jax.jacobian(_f), in_axes=(None, 0))

    # Distribute upper-triangular pairs across devices.
    pairs = list(combinations_with_replacement(range(N_batches), 2))
    n_pairs = len(pairs)
    pairs_per_device = (n_pairs + n_devices - 1) // n_devices
    device_pairs = pairs[rank * pairs_per_device : (rank + 1) * pairs_per_device]
    if len(device_pairs) < pairs_per_device:    # pad so all devices run same iters
        device_pairs += [device_pairs[0]] * (pairs_per_device - len(device_pairs))
    device_pairs = jnp.array(device_pairs)

    if mode == "complex":
        ntk = jnp.zeros((N_batches, N_mc_per_batch, N_batches, N_mc_per_batch, 2, 2))
    else:
        ntk = jnp.zeros((N_batches, N_mc_per_batch, N_batches, N_mc_per_batch))

    def body_fun(k, ntk):
        i, j    = device_pairs[k, 0], device_pairs[k, 1]
        state_i = jax.tree.map(lambda x: x[i], global_state)
        state_j = jax.tree.map(lambda x: x[j], global_state)
        J1      = jacobian_fn(wf.params, state_i)
        J2      = jacobian_fn(wf.params, state_j)
        ntk_ij  = _ntk_contraction(J1, J2, mode, V=V)
        ntk = ntk.at[i, :, j].set(ntk_ij)
        if mode == "complex":
            ntk = ntk.at[j, :, i].set(ntk_ij.transpose(1, 0, 3, 2))
        else:
            ntk = ntk.at[j, :, i].set(ntk_ij.T)
        return ntk

    ntk = jax.lax.fori_loop(0, device_pairs.shape[0], body_fun, ntk)
    ntk = jax.lax.psum(ntk, 'i')
    ntk = rearrange(ntk, 'i a j b ... -> (i a) (j b) ...')
    return ntk


def center_ntk(ntk, weights, state):
    """Subtract row, column, and global means from the NTK (centering step).

    Uses per-system means when `state` is a FoundationState. `ntk` is already
    fully gathered/replicated across devices by this point (see
    ntk_parallel_fn), so reusing grouped_mean here (built for device-sharded
    data, with an internal psum) is still correct: the spurious n_devices
    factor from summing identical replicas cancels between grouped_mean's
    numerator and denominator.
    """
    if isinstance(state, FoundationState):
        if weights is not None:
            raise NotImplementedError(
                "center_ntk: weights + FoundationState is not supported yet."
            )
        system_ids  = jax.lax.all_gather(state.system_ids, 'i', tiled=True)
        K           = state.n_systems
        row_mean    = grouped_mean(ntk, system_ids, K, axis=0, broadcast=True)
        col_mean    = grouped_mean(ntk, system_ids, K, axis=1, broadcast=True)
        global_mean = grouped_mean(row_mean, system_ids, K, axis=1, broadcast=True)
    elif weights is None:
        row_mean    = jnp.mean(ntk, axis=0, keepdims=True)
        col_mean    = jnp.mean(ntk, axis=1, keepdims=True)
        global_mean = jnp.mean(ntk, axis=(0, 1), keepdims=True)
    else:
        w = jax.lax.all_gather(weights, 'i', tiled=True)
        w = jnp.expand_dims(w, axis=tuple(range(1, ntk.ndim - 1)))
        row_mean    = jnp.mean(w[:, None] * ntk, axis=0, keepdims=True)
        col_mean    = jnp.mean(w[None, :] * ntk, axis=1, keepdims=True)
        global_mean = jnp.mean(
            w[:, None] * ntk * w[None, :], axis=(0, 1), keepdims=True
        )

    return ntk - row_mean - col_mean + global_mean


def compute_ntk(state, wf, mode, weights=None, V=None, nbatches=1):
    """Full NTK pipeline: parallel assembly → centering → optional weight scaling."""
    ntk = ntk_parallel_fn(state, wf, nbatches, mode, V=V)
    ntk = center_ntk(ntk, weights, state)

    if weights is not None:
        sqrt_w = jax.lax.all_gather(jnp.sqrt(weights), 'i', tiled=True)
        if mode == "complex":
            ntk = ntk * sqrt_w[:, None, None, None] * sqrt_w[None, :, None, None]
        else:
            ntk = ntk * sqrt_w[:, None] * sqrt_w[None, :]

    return ntk


def center_sr_solution(sr_solution, state, mode, weights):
    """Center the linear-solve output before the VJP step."""
    N_mc = get_n_mc(state)

    if mode == "complex":
        sr_solution = sr_solution.reshape(2, -1).T  # (N_mc, 2)

    if isinstance(state, FoundationState):
        if weights is not None:
            raise NotImplementedError(
                "center_sr_solution: weights + FoundationState is not supported yet."
            )
        system_ids  = jax.lax.all_gather(state.system_ids, 'i', tiled=True)
        mean_a      = grouped_mean(sr_solution, system_ids, state.n_systems, axis=0, broadcast=True)
        sr_solution = (sr_solution - mean_a) / N_mc ** 0.5
    elif weights is not None:
        w = jax.lax.all_gather(weights, 'i', tiled=True)
        if mode == "complex":
            w = w[:, None]
        sr_solution = jnp.sqrt(w) * sr_solution
        mean_a      = jnp.mean(sr_solution, axis=0, keepdims=True)
        sr_solution = (sr_solution - mean_a * w) / N_mc ** 0.5
    else:
        sr_solution = (sr_solution - jnp.mean(sr_solution, axis=0)) / N_mc ** 0.5

    return sr_solution
