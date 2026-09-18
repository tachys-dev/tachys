"""The spectrally-truncated solver used by the real-time dynamics driver.

Anchored against ``linear_solver_cholesky`` (which the SR regression tests pin
numerically) and against ``numpy.linalg.pinv``, so this file does not itself
encode any convention -- it only asserts that the eigendecomposition route
reproduces the one already in use.
"""
import numpy as np
import pytest

import jax
import jax.numpy as jnp

from tachys.optimizer._kernels import (
    _ntk_to_dense,
    _solver_rhs,
    linear_solver_cholesky,
    linear_solver_eigh,
)

M, P = 7, 40          # 2M = 14 < P, so the kernel is full rank and well conditioned
SHIFT = 1e-3


def _random_complex_ntk(seed=3, M=M, P=P):
    """NTK block array (M, M, 2, 2) from a random stacked real Jacobian."""
    rng = np.random.default_rng(seed)
    J = rng.normal(size=(M, 2, P))
    ntk = jnp.asarray(np.einsum('iap,jbp->ijab', J, J) / M)
    eps = jnp.asarray(rng.normal(size=M) + 1j * rng.normal(size=M))
    return ntk, eps, J


def test_ntk_to_dense_is_the_gram_matrix_of_the_stacked_jacobian():
    """(M,M,2,2) -> (2M,2M) must be component-major: [u block; v block].

    This is the single most dangerous line in the solver -- the interleaved
    ordering produces a perfectly finite but wrong answer, because
    ``center_sr_solution`` un-stacks the result with ``reshape(2, -1).T``.
    """
    ntk, _, J = _random_complex_ntk()
    J_full = np.concatenate([J[:, 0, :], J[:, 1, :]], axis=0)   # rows a*M + i
    G = np.asarray(_ntk_to_dense(ntk, "complex"))

    assert G.shape == (2 * M, 2 * M)
    assert np.allclose(G, J_full @ J_full.T / M, atol=1e-14)
    assert np.allclose(G, G.T, atol=1e-14)
    # blocks are [[A, B], [B.T, C]] with A/B/C read off the last two axes
    assert np.allclose(G[:M, :M], np.asarray(ntk[..., 0, 0]), atol=1e-14)
    assert np.allclose(G[:M, M:], np.asarray(ntk[..., 0, 1]), atol=1e-14)
    assert np.allclose(G[M:, M:], np.asarray(ntk[..., 1, 1]), atol=1e-14)

    # and the interleaved (sample-major, row index i*2 + a) alternative really is
    # different, so this is not a distinction without a difference
    interleaved = np.asarray(jnp.transpose(ntk, (0, 2, 1, 3)).reshape(2 * M, 2 * M))
    assert not np.allclose(interleaved, G)

    # mode="real" passes through untouched
    assert _ntk_to_dense(ntk[..., 0, 0], "real").shape == (M, M)


def test_solver_rhs_negates_the_imaginary_part():
    """The block system is [[A,B],[B.T,C]] [u;v] = [Re eps; -Im eps]."""
    _, eps, _ = _random_complex_ntk()
    b = _solver_rhs(eps, "complex")
    assert np.allclose(b[:M], np.asarray(eps.real))
    assert np.allclose(b[M:], -np.asarray(eps.imag))


def test_cholesky_solves_the_block_system_we_claim_it_does():
    """Pins the convention the eigh solver has to reproduce."""
    ntk, eps, _ = _random_complex_ntk()
    sol = np.asarray(linear_solver_cholesky(ntk, eps, SHIFT, mode="complex"))
    u, v = sol[:M], sol[M:]
    A = np.asarray(ntk[..., 0, 0]) + SHIFT * np.eye(M)
    B = np.asarray(ntk[..., 0, 1])
    C = np.asarray(ntk[..., 1, 1]) + SHIFT * np.eye(M)
    assert np.allclose(A @ u + B @ v, np.asarray(eps.real), atol=1e-12)
    assert np.allclose(B.T @ u + C @ v, -np.asarray(eps.imag), atol=1e-12)


@pytest.mark.parametrize("mode", ["complex", "real"])
def test_eigh_matches_cholesky_when_well_conditioned(mode):
    """With a negligible cutoff the two solvers must agree to roundoff."""
    ntk, eps, J = _random_complex_ntk()
    if mode == "real":
        ntk = ntk[..., 0, 0]
    chol = linear_solver_cholesky(ntk, eps, SHIFT, mode=mode)
    eigh = linear_solver_eigh(ntk, eps, SHIFT, mode=mode, rcond=1e-14)
    assert np.allclose(np.asarray(chol), np.asarray(eigh), atol=1e-11)


def test_eigh_matches_pinv_on_a_rank_deficient_kernel():
    """2M > P makes the kernel singular -- exactly the dynamics regime, where
    Cholesky has nothing to compare against."""
    ntk, eps, _ = _random_complex_ntk(seed=5, M=6, P=3)
    G = np.asarray(_ntk_to_dense(ntk, "complex"))
    rhs = np.concatenate([np.asarray(eps.real), -np.asarray(eps.imag)])
    ref = np.linalg.pinv(G, rcond=1e-8, hermitian=True) @ rhs
    got = np.asarray(linear_solver_eigh(ntk, eps, 0.0, mode="complex", rcond=1e-8))
    assert np.allclose(ref, got, atol=1e-12)


def test_eigh_discards_directions_below_the_cutoff():
    """Raising rcond past an eigenvalue must remove that direction entirely."""
    lam = jnp.asarray([1.0, 1e-3, 1e-9])
    Q = jnp.eye(3)
    ntk = Q @ jnp.diag(lam) @ Q.T
    eps = jnp.asarray([1.0, 1.0, 1.0], dtype=jnp.complex128)

    keep_all = np.asarray(linear_solver_eigh(ntk, eps, 0.0, mode="real", rcond=1e-12))
    assert np.allclose(keep_all, [1.0, 1e3, 1e9])            # 1/lambda on every mode

    keep_two = np.asarray(linear_solver_eigh(ntk, eps, 0.0, mode="real", rcond=1e-6))
    assert np.allclose(keep_two, [1.0, 1e3, 0.0])            # the 1e-9 mode is gone

    keep_one = np.asarray(linear_solver_eigh(ntk, eps, 0.0, mode="real", rcond=1e-2))
    assert np.allclose(keep_one, [1.0, 0.0, 0.0])

    # atol is an absolute floor on the same cutoff
    assert np.allclose(
        np.asarray(linear_solver_eigh(ntk, eps, 0.0, mode="real", rcond=0.0, atol=1e-2)),
        [1.0, 0.0, 0.0])


def test_eigh_rejects_negative_eigenvalues():
    """A negative eigenvalue is roundoff on a signal-free direction; inverting it
    would flip the update along that direction and amplify it by 1/|lambda|."""
    ntk = jnp.diag(jnp.asarray([1.0, -1e-12]))
    eps = jnp.asarray([1.0, 1.0], dtype=jnp.complex128)
    sol = np.asarray(linear_solver_eigh(ntk, eps, 0.0, mode="real", rcond=1e-14))
    assert np.allclose(sol, [1.0, 0.0])


def test_eigh_diag_shift_does_not_disable_the_truncation():
    """The keep mask reads the raw spectrum, so a large shift must not silently
    turn the cutoff into a no-op."""
    ntk = jnp.diag(jnp.asarray([1.0, 1e-9]))
    eps = jnp.asarray([1.0, 1.0], dtype=jnp.complex128)
    sol = np.asarray(linear_solver_eigh(ntk, eps, 1e-2, mode="real", rcond=1e-6))
    assert sol[1] == 0.0                       # truncated despite shift >> rcond*lam_max
    assert np.isclose(sol[0], 1.0 / (1.0 + 1e-2))


def test_eigh_degrades_to_a_no_op_on_non_finite_input():
    """Same guard as the Cholesky path: never NaN-poison the parameters."""
    ntk, eps, _ = _random_complex_ntk()
    ntk = ntk.at[0, 0, 0, 0].set(jnp.nan)
    assert np.all(np.asarray(linear_solver_eigh(ntk, eps, 0.0, mode="complex")) == 0.0)


def test_eigh_returns_zero_for_an_all_junk_spectrum():
    ntk, eps, _ = _random_complex_ntk()
    sol = linear_solver_eigh(jnp.zeros_like(ntk), eps, 0.0, mode="complex")
    assert np.all(np.asarray(sol) == 0.0)


def test_eigh_is_jittable_with_traced_thresholds():
    ntk, eps, _ = _random_complex_ntk()
    f = jax.jit(lambda n, e, d, r: linear_solver_eigh(n, e, d, mode="complex", rcond=r))
    got = f(ntk, eps, 0.0, jnp.asarray([1e-8]))
    ref = linear_solver_eigh(ntk, eps, 0.0, mode="complex", rcond=1e-8)
    assert np.allclose(np.asarray(got), np.asarray(ref))


def test_eigh_rejects_unknown_mode():
    ntk, eps, _ = _random_complex_ntk()
    with pytest.raises(ValueError, match="mode must be"):
        linear_solver_eigh(ntk, eps, 0.0, mode="quaternion")
