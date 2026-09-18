"""Exact checks on the single Lanczos step of tachys.experimental.lanczos.

Everything here is deterministic: the Appendix C algebra is checked against
explicit linear algebra on a random Hermitian matrix, and the Lanczos-step
wavefunction against ``(1 + alpha H) psi_G`` built as a dense matrix-vector
product over the full Hilbert space of an 8-site chain.  No Monte Carlo, so no
statistical tolerances -- the Monte Carlo estimators are the *same* expressions
evaluated on samples, and what could go wrong in them (a wrong conjugation, a
wrong normalisation, the wrong local energy) shows up here at 1e-14.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tachys.experimental.lanczos import (
    _jackknife, c5_residual, chi_estimator, lanczos_chi, lanczos_energy, lanczos_moments,
    lanczos_step, lanczos_wavefunction, measure_energy, measure_lanczos, optimal_alpha,
    stationarity_coefficients,
)
from tachys.lattice.ansatz.rbm import SpinRBM
from tachys.lattice.exact_diag import spins_hilbert_space
from tachys.lattice.lattice_database import chain
from tachys.lattice.operator.local_estimator import local_estimator
from tachys.lattice.spins.hamiltonians.ising_transverse_field import (
    ising_transverse_field_hamiltonian,
)
from tachys.lattice.spins.hamiltonians.heisenberg import heisenberg_square_pbc
from tachys.lattice.spins.spin_action import SpinFlip
from tachys.lattice.spins.spin_state import SpinState
from tachys.wavefunction import WaveFunction

L = 8
N = 2 ** L
ALPHAS = (-0.31, -0.05, 0.02)


# ---------------------------------------------------------------- exact setup

def _pack(state):
    bits = (np.asarray(state.spins) + 1) // 2
    return (bits * 2 ** np.arange(state.spins.shape[-1])).sum(axis=-1)


def _dense_hamiltonian(H, state, idx):
    """Dense matrix of a tachys operator over an enumerated basis.

    Same construction as ``exact_diag.exact_diag``, kept dense and un-factored
    here so the test does not depend on that function's sparse/eigensolver path.
    """
    res = H(state)
    dim = idx.size
    M = np.zeros((dim, dim))
    np.add.at(M, (idx, idx), np.asarray(res.diagonal.matrix_element).sum(0))
    cols = np.asarray(_pack(res.offdiagonal.connected_states))
    vals = np.asarray(res.offdiagonal.mask * res.offdiagonal.matrix_element)
    for t in range(cols.shape[0]):
        np.add.at(M, (idx, cols[t]), vals[t])
    assert np.allclose(M, M.T), "the test Hamiltonian must come out symmetric"
    return M


@pytest.fixture(scope="module", params=[False, True], ids=["real", "complex"])
def setup(request):
    """8-site TFIM chain, a random RBM guiding state, and the exact objects.

    The ``complex`` variant matters: it is the only thing that distinguishes
    ``<x|H|psi>/<x|psi>`` (what builds ``psi_alpha``) from its conjugate, and
    ``<psi_alpha|psi_G>`` from ``<psi_G|psi_alpha>``.
    """
    lat = chain(L, pbc=True)
    H = ising_transverse_field_hamiltonian(lat, nn=[((1, 0), 1.0)], h=1.0)

    basis = spins_hilbert_space(lat.Ns)
    state = SpinState(spins=jnp.asarray(basis, dtype=jnp.int8), lattice=lat)
    idx = _pack(state)
    assert np.array_equal(idx, np.arange(N)), "basis must be in packed order"
    M = _dense_hamiltonian(H, state, idx)

    model = SpinRBM(hidden_units=L, dtype=jnp.float64, complex=request.param)
    dummy = SpinState(spins=jnp.ones((1, L)), lattice=lat)
    wf = WaveFunction(params=model.init(jax.random.key(3), dummy), apply_fn=model.apply)

    log_psi = np.asarray(wf.apply_fn(wf.params, state)).astype(complex)
    # Amplitudes up to a common constant; the constant cancels from every ratio
    # below, and subtracting the max keeps exp() in range.
    psi = np.exp(log_psi - log_psi.real.max())
    return dict(H=H, lat=lat, state=state, M=M, wf=wf, log_psi=log_psi, psi=psi)


def _moments(M, psi, n):
    """h_n = <psi|H^n|psi> / <psi|psi>, exactly."""
    w = psi.copy()
    for _ in range(n):
        w = M @ w
    return float(np.real(np.vdot(psi, w) / np.vdot(psi, psi)))


# ------------------------------------------------------- the LS wavefunction

@pytest.mark.parametrize("alpha", ALPHAS)
def test_lanczos_wavefunction_amplitudes(setup, alpha):
    """exp(log psi_alpha) == (1 + alpha H) psi_G on every basis state."""
    wf, state, M, psi, log_psi = (setup[k] for k in ("wf", "state", "M", "psi", "log_psi"))
    wf_alpha = lanczos_wavefunction(wf, setup["H"], alpha)

    got = np.asarray(wf_alpha.apply_fn(wf_alpha.params, state)).astype(complex)
    got = np.exp(got - log_psi.real.max())
    expected = psi + alpha * (M @ psi)

    assert np.allclose(got, expected, rtol=1e-11, atol=1e-11 * np.abs(psi).max())


@pytest.mark.parametrize("alpha", ALPHAS)
def test_local_estimator_is_the_amplitude_ratio(setup, alpha):
    """local_estimator returns <x|H|psi>/<x|psi>, the ratio psi_alpha needs.

    For a complex guiding state the conjugate convention <psi|H|x>/<psi|x> --
    which is how Appendix C writes it -- differs by a complex conjugation, and
    using it would build the wrong wavefunction.
    """
    wf, state, M, psi = (setup[k] for k in ("wf", "state", "M", "psi"))
    log_amps = wf.apply_fn(wf.params, state).astype(jnp.complex128)
    e_G = np.asarray(local_estimator(setup["H"], state, wf, log_amps))
    assert np.allclose(e_G, (M @ psi) / psi, rtol=1e-11, atol=1e-11)


# ------------------------------------------------------------ Appendix C algebra

@pytest.mark.parametrize("alpha", ALPHAS)
def test_c3_and_chi_match_explicit_vectors(setup, alpha):
    """(C3) and chi(alpha) reproduce the ratios formed from the explicit vectors."""
    M, psi = setup["M"], setup["psi"]
    h1, h2, h3 = (_moments(M, psi, n) for n in (1, 2, 3))

    psi_a = psi + alpha * (M @ psi)
    norm = np.vdot(psi_a, psi_a)
    e_exact = np.real(np.vdot(psi_a, M @ psi_a) / norm)
    chi_exact = np.real(np.vdot(psi_a, psi) / norm)   # <psi_alpha|(1+aH)^-1|psi_alpha>

    assert lanczos_energy(alpha, h1, h2, h3) == pytest.approx(e_exact, rel=1e-10)
    assert lanczos_chi(alpha, h1, h2) == pytest.approx(chi_exact, rel=1e-10)


@pytest.mark.parametrize("alpha", ALPHAS)
def test_moment_inversion_round_trip(setup, alpha):
    """Eqs. (4)-(5) invert chi and E(alpha) back to h2 and h3."""
    M, psi = setup["M"], setup["psi"]
    h1, h2, h3 = (_moments(M, psi, n) for n in (1, 2, 3))
    h2_out, h3_out = lanczos_moments(
        alpha, h1, lanczos_chi(alpha, h1, h2), lanczos_energy(alpha, h1, h2, h3))
    assert h2_out == pytest.approx(h2, rel=1e-8)
    assert h3_out == pytest.approx(h3, rel=1e-6)


def test_optimal_alpha_matches_brute_force(setup):
    """(C4) is the minimiser of (C3): negative, below h1, and C5-consistent."""
    M, psi = setup["M"], setup["psi"]
    h1, h2, h3 = (_moments(M, psi, n) for n in (1, 2, 3))
    alpha_star, energy_star, roots = optimal_alpha(h1, h2, h3)

    grid = np.linspace(-2.0, 2.0, 400001)
    e_grid = lanczos_energy(grid, h1, h2, h3)
    assert alpha_star == pytest.approx(grid[np.argmin(e_grid)], abs=2e-5)
    assert energy_star == pytest.approx(e_grid.min(), rel=1e-9)

    # The sign of alpha* is state dependent (it follows -1/E_1 of the dominant
    # admixture), so the invariants to assert are Vieta's and the variational
    # bound -- NOT a sign. dE/dalpha|_0 = 2 Var > 0 only rules out alpha* = 0.
    a, b, c = stationarity_coefficients(h1, h2, h3)
    assert roots[0] * roots[1] == pytest.approx(c / a, rel=1e-9)
    assert roots[0] + roots[1] == pytest.approx(-b / a, rel=1e-9)
    assert np.sign(roots[0] * roots[1]) == np.sign(a)     # because c = Var > 0
    assert alpha_star != 0.0
    assert energy_star < h1                               # alpha = 0 is in the family
    assert lanczos_energy(roots[1] if roots[1] != alpha_star else roots[0],
                          h1, h2, h3) > energy_star

    # Eq. (C5) is an identity at alpha*, and only there.
    assert c5_residual(alpha_star, lanczos_chi(alpha_star, h1, h2), energy_star) == pytest.approx(0.0, abs=1e-12)
    off = 0.5 * alpha_star
    assert abs(c5_residual(off, lanczos_chi(off, h1, h2), lanczos_energy(off, h1, h2, h3))) > 1e-8


def test_optimal_alpha_degenerate_inputs():
    """An eigenstate (zero variance) and unusable moments are reported, not faked."""
    # h1 = h2 = h3 = E: every coefficient of the stationarity quadratic vanishes.
    a_s, e_s, _ = optimal_alpha(-1.0, 1.0, -1.0)
    assert np.isnan(a_s) and np.isnan(e_s)
    a_s, _, _ = optimal_alpha(float("nan"), 1.0, 1.0)
    assert np.isnan(a_s)


# ------------------------------------------------------------- the chi estimator

@pytest.mark.parametrize("alpha", ALPHAS)
def test_chi_estimator_equals_local_energy_form(setup, alpha):
    """mean(psi_G/psi_alpha) == mean(1/(1 + alpha e_G)), term by term.

    The module samples chi in the amplitude-ratio form, which costs one forward
    pass instead of a whole local-energy evaluation; this pins the two forms
    together on a batch of real configurations.
    """
    wf, H = setup["wf"], setup["H"]
    lat = setup["lat"]
    key = jax.random.key(7)
    spins = jnp.sign(jax.random.normal(key, (32, L))).astype(jnp.int8)
    state = SpinState(spins=spins, lattice=lat)

    wf_alpha = lanczos_wavefunction(wf, H, alpha)
    log_amps_alpha = wf_alpha.apply_fn(wf_alpha.params, state).astype(jnp.complex128)
    chi, log_amps_G = chi_estimator(wf, state, log_amps_alpha)

    e_G = local_estimator(H, state, wf, log_amps_G)
    assert complex(chi) == pytest.approx(complex(jnp.mean(1.0 / (1.0 + alpha * e_G))), rel=1e-10)


def test_chi_estimator_is_the_right_overlap(setup):
    """chi is <psi_alpha|psi_G>/<psi_alpha|psi_alpha>, bra conjugated on psi_alpha.

    Evaluated over the *whole* Hilbert space with |psi_alpha|^2 weights, the
    Monte Carlo average becomes the exact ratio, which is what makes this a test
    of the conjugation convention rather than of the sampler.
    """
    wf, state, M, psi = (setup[k] for k in ("wf", "state", "M", "psi"))
    alpha = -0.05
    wf_alpha = lanczos_wavefunction(wf, setup["H"], alpha)
    log_amps_alpha = wf_alpha.apply_fn(wf_alpha.params, state).astype(jnp.complex128)
    log_amps_G = wf.apply_fn(wf.params, state).astype(jnp.complex128)

    ratio = np.asarray(jnp.exp(log_amps_G - log_amps_alpha))
    psi_a = np.asarray(np.exp(np.asarray(log_amps_alpha) - np.asarray(log_amps_alpha).real.max()))
    weights = np.abs(psi_a) ** 2
    chi_sampled = np.sum(weights * ratio) / np.sum(weights)

    psi_a_exact = psi + alpha * (M @ psi)
    chi_exact = np.vdot(psi_a_exact, psi) / np.vdot(psi_a_exact, psi_a_exact)
    assert chi_sampled == pytest.approx(complex(chi_exact), rel=1e-10)


# ------------------------------------------------- the two-level limit, exactly

@pytest.mark.parametrize(
    "E0, E1",
    [(-10.0, -9.0),    # both negative: alpha* = -1/E1 > 0  (the usual lattice-model case)
     (-10.0, 9.0),     # contaminant above zero: alpha* < 0
     (1.0, 2.0),       # both positive: alpha* < 0 even though a > 0
     (-2.0, -1.0)],
)
def test_two_level_stationary_points_are_inverse_eigenvalues(E0, E1):
    """For a two-level guiding state the LS step is exact, and (C4) knows it.

    With weight ``(1-p, p)`` on ``(E_0, E_1)`` the stationarity quadratic
    factorises into ``alpha = -1/E_0`` and ``alpha = -1/E_1``: the second
    annihilates the contaminant, so ``E(alpha*) = E_0`` exactly and one Lanczos
    step reaches the ground state.  This is the sharpest statement of where
    ``alpha*`` sits and of what sets its sign -- neither root's sign is fixed by
    ``dE/dalpha|_0 = 2 Var > 0``.
    """
    p = 0.2
    w = np.array([1.0 - p, p])
    E = np.array([E0, E1])
    h1, h2, h3 = ((w * E ** n).sum() for n in (1, 2, 3))

    alpha_star, energy_star, roots = optimal_alpha(h1, h2, h3)
    assert sorted(roots) == pytest.approx(sorted([-1.0 / E0, -1.0 / E1]), rel=1e-9)
    assert alpha_star == pytest.approx(-1.0 / E1, rel=1e-9)
    assert energy_star == pytest.approx(E0, rel=1e-9)
    # ... and the other root is the maximum, sitting at the excited level.
    other = roots[0] if roots[1] == pytest.approx(alpha_star) else roots[1]
    assert lanczos_energy(other, h1, h2, h3) == pytest.approx(E1, rel=1e-9)
    # chi vanishes at the orthogonality point alpha = -1/h1, always ...
    assert lanczos_chi(-1.0 / h1, h1, h2) == pytest.approx(0.0, abs=1e-12)
    # ... and that point sits *between* the two stationary points whenever E_0
    # and E_1 share a sign, which is what makes a probe iterated towards alpha*
    # step over it. (When the two levels straddle zero, so does the pole of
    # E -> -1/E, and -1/h1 lands outside the interval instead.)
    if E0 * E1 > 0:
        assert min(roots) < -1.0 / h1 < max(roots)


# ------------------------------------------------------------- the noise guards

def test_optimal_alpha_rejects_negative_derived_variance():
    """h2 <= h1^2 means (C3) has poles, not a minimum: nan, not the local max.

    A negative *derived* variance is what finite statistics produce for a
    well-optimised psi_G, and it is the only way the nan path can be reached --
    see the discriminant identity below.  Returning "the root with the lower E"
    there would hand back a local maximum sitting between two zeros of the norm.
    """
    h1 = -10.0
    for c in (-1e-3, -1e-9, 0.0):
        h2 = h1 ** 2 + c
        a_s, e_s, roots = optimal_alpha(h1, h2, -1000.0)
        assert np.isnan(a_s) and np.isnan(e_s) and all(np.isnan(r) for r in roots)
    # ... and it is accepted again as soon as the variance is positive.
    assert np.isfinite(optimal_alpha(h1, h1 ** 2 + 1e-6, -1000.0)[0])


@pytest.mark.parametrize("seed", range(5))
def test_discriminant_is_positive_whenever_the_variance_is(seed):
    """b^2 - 4ac == (h3 - 3 h1 h2 + 2 h1^3)^2 + 4 (h2 - h1^2)^3 >= 4 Var^3.

    So no amount of noise in h3 can make (C4) complex while Var > 0: the nan
    path documented in ``optimal_alpha`` is reachable only through a negative
    derived variance, and a nan must not be blamed on a noisy h3.
    """
    rng = np.random.default_rng(seed)
    h1, h2, h3 = rng.normal() * 5, abs(rng.normal()) * 5, rng.normal() * 50
    if h2 <= h1 ** 2:
        h2 = h1 ** 2 + abs(rng.normal())
    a, b, c = stationarity_coefficients(h1, h2, h3)
    assert b * b - 4 * a * c == pytest.approx(
        (h3 - 3 * h1 * h2 + 2 * h1 ** 3) ** 2 + 4 * (h2 - h1 ** 2) ** 3, rel=1e-9)
    assert b * b - 4 * a * c >= 4 * c ** 3


def test_c5_residual_vanishes_at_both_roots_and_ignores_bias():
    """C5 measures "the probe is stationary", nothing about estimator bias.

    It is alpha^2 q(alpha) / (...) with q the quadratic (C4) solves, and h2, h3
    are exact inversions of the same chi and E, so a biased chi moves alpha*
    while leaving the residual at exactly zero -- and it vanishes at the
    maximum root too.  Pinned here so the docstring's modest claim stays true.
    """
    h1, h2, h3 = -9.95, 100.0, -1005.0
    _, _, roots = optimal_alpha(h1, h2, h3)
    for r in roots:
        assert c5_residual(r, lanczos_chi(r, h1, h2), lanczos_energy(r, h1, h2, h3)) \
            == pytest.approx(0.0, abs=1e-12)

    # A 1% bias in chi at a self-consistent probe: alpha* moves, residual stays 0.
    alpha = roots[0]
    chi_biased = 1.01 * lanczos_chi(alpha, h1, h2)
    energy = lanczos_energy(alpha, h1, h2, h3)
    h2_b, h3_b = lanczos_moments(alpha, h1, chi_biased, energy)
    alpha_biased = optimal_alpha(h1, h2_b, h3_b)[0]
    assert alpha_biased != pytest.approx(alpha, rel=1e-6)
    res_old = abs(c5_residual(alpha, chi_biased, energy))
    assert res_old > 1e-12
    # ... but evaluated self-consistently with the biased moments it collapses
    # back to zero, so a run at a converged probe reports residual ~ 0 whatever
    # the bias. (Not exactly zero in floating point: the residual is
    # alpha^2 q(alpha)/(...) and q has a large slope at its root for h3 ~ -1e3.)
    res_new = abs(c5_residual(alpha_biased, lanczos_chi(alpha_biased, h1, h2_b),
                              lanczos_energy(alpha_biased, h1, h2_b, h3_b)))
    assert res_new < 1e-8
    assert res_new < 1e-3 * res_old


def test_jackknife_h1_error_is_monotonic():
    """A ten-fold larger h1 error must not shrink alpha_star_err.

    The naive central difference across the whole +-h1_err interval can wrap
    around the pole where a = h1 h3 - h2^2 changes sign and come back smaller,
    quoting a confident bar exactly where alpha* is unbounded.  The module takes
    the derivative at a small step and returns inf when the interval spans the
    pole; either way the bar may not decrease.
    """
    h1, alpha = -4.4001, -0.2
    h2, h3 = h1 ** 2 + 0.5569, -89.734
    chi = np.full(64, lanczos_chi(alpha, h1, h2))
    energy = np.full(64, lanczos_energy(alpha, h1, h2, h3))
    rng = np.random.default_rng(0)
    chi = chi * (1 + 1e-6 * rng.normal(size=64))
    energy = energy * (1 + 1e-6 * rng.normal(size=64))

    errs = [_jackknife(alpha, h1, e, energy, chi, n_blocks=8)["alpha_star"]
            for e in (1e-3, 1e-2, 1e-1, 1.0)]
    assert all(np.isfinite(e) or np.isinf(e) for e in errs)
    assert all(b >= a or np.isinf(b) for a, b in zip(errs, errs[1:])), errs


# -------------------------------------------------------------- API guardrails

def test_lanczos_wavefunction_records_alpha_and_measure_lanczos_checks_it(setup):
    """A prebuilt wf_alpha at the wrong alpha must raise, not silently corrupt.

    The alpha is baked into the closure and is unrecoverable from the
    WaveFunction, so ``lanczos_wavefunction`` records it on the closure and
    ``measure_lanczos`` refuses a mismatch -- otherwise the chain is sampled
    from one LS state while Eqs. (4)-(5) are solved for another.
    """
    wf, H, lat = setup["wf"], setup["H"], setup["lat"]
    wf_alpha = lanczos_wavefunction(wf, H, -0.05)
    assert wf_alpha.apply_fn.lanczos_alpha == -0.05

    spins = jnp.ones((4, L), dtype=jnp.int8)
    state = SpinState(spins=spins, lattice=lat)
    with pytest.raises(ValueError, match="built at alpha"):
        measure_lanczos(jax.random.key(0), wf, H, state, SpinFlip(), 4, 1,
                        alpha=-0.4, h1=-1.0, wf_alpha=wf_alpha, log_every=0)
    with pytest.raises(ValueError, match="must come from lanczos_wavefunction"):
        measure_lanczos(jax.random.key(0), wf, H, state, SpinFlip(), 4, 1,
                        alpha=-0.05, h1=-1.0, wf_alpha=wf, log_every=0)


def test_lanczos_wavefunction_rejects_per_chain_coupling(setup):
    """A foundation-model Hamiltonian (2-D coupling) must be refused up front.

    apply_fn is called on resized and permuted walker sub-batches, so a
    coupling indexed by walker position would stop matching its rows -- a crash
    in ``sample`` and a silently wrong E(alpha) in a direct
    ``compute_expectation``. Better to refuse than to be subtly wrong.
    """
    from tachys.lattice.foundation.operators import combine_systems
    H_comb = combine_systems([heisenberg_square_pbc(2, J=1.0),
                              heisenberg_square_pbc(2, J=2.0)], 8)
    with pytest.raises(ValueError, match="per-chain coupling"):
        lanczos_wavefunction(setup["wf"], H_comb, -0.05)


def test_degenerate_loop_and_step_counts_raise(setup):
    """n_iter=0 and N_steps=0 are rejected before any sampling is paid for."""
    wf, H, lat = setup["wf"], setup["H"], setup["lat"]
    state = SpinState(spins=jnp.ones((4, L), dtype=jnp.int8), lattice=lat)
    args = (jax.random.key(0), wf, H, state, SpinFlip(), 4)
    with pytest.raises(ValueError, match="n_iter must be at least 1"):
        lanczos_step(*args, N_steps=1, n_iter=0, h1=-1.0)
    with pytest.raises(ValueError, match="N_steps must be at least 1"):
        lanczos_step(*args, N_steps=0, n_iter=1, h1=-1.0)
    with pytest.raises(ValueError, match="N_steps must be at least 1"):
        measure_lanczos(*args, N_steps=0, alpha=-0.05, h1=-1.0)
    with pytest.raises(ValueError, match="N_steps must be at least 1"):
        measure_energy(*args, N_steps=0)
    with pytest.raises(ValueError, match="alpha must be non-zero"):
        measure_lanczos(*args, N_steps=1, alpha=0.0, h1=-1.0)


# ------------------------------------------------------- the printed diagnostics

def _dummy_result(**over):
    """A LanczosResult with plausible fields, for exercising _probe_warnings."""
    from tachys.experimental.lanczos import LanczosResult
    base = dict(
        alpha=0.02, h1=-10.0, h1_err=1e-3, h2=110.0, h2_err=1e-2, h3=-1200.0, h3_err=1.0,
        energy=-10.05, energy_err=1e-3, chi=0.9, chi_err=1e-4,
        variance_alpha=5.0, variance_alpha_err=0.1, variance_guiding=10.0,
        variance_direct=10.0, variance_mismatch=0.0,
        alpha_star=0.05, alpha_star_err=1e-3, energy_star=-10.2, energy_star_err=1e-3,
        c5_residual=1e-9, imag_energy=0.0, imag_chi=0.0, acceptance=0.5,
        roots=(0.05, 0.01), history={},
    )
    base.update(over)
    return LanczosResult(**base)


@pytest.mark.parametrize("over, expect", [
    ({}, None),
    ({"variance_guiding": -1e-3, "alpha_star": float("nan")}, "no minimum"),
    ({"alpha": 0.0999}, "orthogonality point"),          # -1/h1 = 0.1
    ({"chi": 1e-3}, "ill-conditioned"),
    ({"energy_star": -9.0}, "above h1"),                 # above h1 = -10
    ({"energy_star": -20.0}, "exceeds sqrt(Var_G)"),     # gain 10 > sqrt(10)
    ({"variance_guiding": 10.0, "variance_direct": 12.0, "variance_mismatch": -2.0},
     "independent estimators"),
])
def test_probe_warnings_fire_on_the_documented_pathologies(over, expect):
    """Every warning branch in ``_probe_warnings`` is reachable and specific.

    These are the diagnostics the Appendix C algebra cannot raise for itself --
    a negative derived variance, a probe on the orthogonality point, a small
    ``|chi|``, an impossible ``E(alpha*)``, an implausible gain, and a
    ``chi``-vs-direct variance disagreement -- so each needs pinning.
    """
    from tachys.experimental.lanczos import _probe_warnings
    msgs = " ".join(_probe_warnings(_dummy_result(**over)))
    if expect is None:
        assert msgs == "", msgs
    else:
        assert expect in msgs, msgs
