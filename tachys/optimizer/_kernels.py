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
        return u

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
        return jnp.concatenate([u, v], axis=-1)

    else:
        raise ValueError("mode must be 'real' or 'complex'")


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
    global_state = jax.lax.all_gather(state, 'i')  # config: (n_devices, N_mc_local, N)

    if nbatches > 1:
        global_state = jax.tree.map(
            lambda x: rearrange(
                x, 'P (nbatches Nmc) ... -> (P nbatches) Nmc ...', nbatches=nbatches
            ),
            global_state,
        )

    N_batches      = global_state.config.shape[0]
    N_mc_per_batch = global_state.config.shape[1]

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


def center_ntk(ntk, weights):
    """Subtract row, column, and global means from the NTK (centering step)."""
    if weights is None:
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
    ntk = center_ntk(ntk, weights)

    if weights is not None:
        sqrt_w = jax.lax.all_gather(jnp.sqrt(weights), 'i', tiled=True)
        if mode == "complex":
            ntk = ntk * sqrt_w[:, None, None, None] * sqrt_w[None, :, None, None]
        else:
            ntk = ntk * sqrt_w[:, None] * sqrt_w[None, :]

    return ntk


def center_sr_solution(sr_solution, state, mode, weights):
    """Center the linear-solve output before the VJP step."""
    N_mc = state.config.shape[0] * n_devices

    if mode == "complex":
        sr_solution = sr_solution.reshape(2, -1).T  # (N_mc, 2)

    if weights is not None:
        w = jax.lax.all_gather(weights, 'i', tiled=True)
        if mode == "complex":
            w = w[:, None]
        sr_solution = jnp.sqrt(w) * sr_solution
        mean_a      = jnp.mean(sr_solution, axis=0, keepdims=True)
        sr_solution = (sr_solution - mean_a * w) / N_mc ** 0.5
    else:
        sr_solution = (sr_solution - jnp.mean(sr_solution, axis=0)) / N_mc ** 0.5

    return sr_solution
