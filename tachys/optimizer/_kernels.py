"""Low-level kernels for SR-family optimizers.

Ported from the jaxvmc reference implementation; adapted for tachys naming
conventions (state / wf instead of lattice / vstate).
"""

from itertools import combinations_with_replacement

from einops import rearrange, einsum
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.scipy.linalg import solve_triangular

from tachys.parallel import mesh, n_devices, rank


# ─── Parallel utility ─────────────────────────────────────────────────────────

def hard_shard(x):
    """Slice a globally-gathered array back to this device's local chunk."""
    chunk = x.shape[0] // n_devices
    return x[rank * chunk : (rank + 1) * chunk]


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

def _jacobian_batch(wf, batch_state, mode):
    """Per-sample Jacobian via jax.grad + ravel_pytree.

    Returns:
        real    mode: (N_mc, N_params) real array
        complex mode: (N_mc, 2, N_params) real array — axis 1 is [J_re, J_im]
    """
    f_real = lambda params, s: jnp.squeeze(wf.apply_fn(params, s).real)
    J_re = jax.vmap(jax.grad(f_real), in_axes=(None, 0))(wf.params, batch_state)
    J_re = jax.vmap(lambda x: ravel_pytree(x)[0], in_axes=0)(J_re)  # (N_mc, N_params)

    if mode == "complex":
        f_imag = lambda params, s: jnp.squeeze(wf.apply_fn(params, s).imag)
        J_im = jax.vmap(jax.grad(f_imag), in_axes=(None, 0))(wf.params, batch_state)
        J_im = jax.vmap(lambda x: ravel_pytree(x)[0], in_axes=0)(J_im)
        return jnp.stack([J_re, J_im], axis=1)  # (N_mc, 2, N_params)

    return J_re  # (N_mc, N_params)


def _ntk_contraction(J1, J2, mode, V_flat=None):
    """Contract flat Jacobian matrices into the NTK block for a pair of batches.

    J1, J2  : (N_i, N_params)       [real]
              (N_i, 2, N_params)    [complex]
    V_flat  : (N_params,) optional diagonal preconditioner (MARCH)

    Returns  : (N_i, N_j)           [real]
               (N_i, N_j, 2, 2)    [complex]
    """
    if V_flat is not None:
        J1 = J1 * (1.0 / (V_flat ** 0.5 + 1e-8))  # scale along last axis

    if mode == "complex":
        return einsum(J1, J2, 'i a p, k b p -> i k a b')
    else:
        return J1 @ J2.T


def ntk_parallel_fn(state, wf, nbatches, mode, V=None):
    """Assemble the full NTK matrix using distributed pair-wise Jacobian contractions.

    Each device computes a subset of (batch_i, batch_j) pairs; contributions
    are summed via psum to yield the (N_mc × N_mc) NTK.
    """
    global_state = jax.lax.all_gather(state, 'i')  # spins: (n_devices, N_mc_local, N)

    if nbatches > 1:
        global_state = jax.tree.map(
            lambda x: rearrange(
                x, 'P (nbatches Nmc) ... -> (P nbatches) Nmc ...', nbatches=nbatches
            ),
            global_state,
        )

    N_batches      = global_state.spins.shape[0]
    N_mc_per_batch = global_state.spins.shape[1]

    V_flat = ravel_pytree(V)[0] if V is not None else None

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
        J1      = _jacobian_batch(wf, state_i, mode)
        J2      = _jacobian_batch(wf, state_j, mode)
        ntk_ij  = _ntk_contraction(J1, J2, mode, V_flat=V_flat)
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
    N_mc = state.spins.shape[0] * n_devices

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
