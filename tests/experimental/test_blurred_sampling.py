"""Tests for tachys.experimental.blurred_sampling.

The load-bearing test is `test_weight_formula_exact`: on a lattice small enough to
enumerate, the blurred density p_blur and the blur weights w are both known in
closed form, so `w * p_blur / p` must be exactly constant with no Monte Carlo
noise anywhere. Everything else (unbiasedness, the q=0 reduction, the proposal
distribution) is pinned on top of that.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tachys.experimental.blurred_sampling import (
    BlurredEstimator, _split_result, blur_states, blurred_local_estimator,
    compute_expectation_blurred,
)
from tachys.lattice.ansatz.rbm import SpinRBM
from tachys.lattice.bond_exchange import BondExchange
from tachys.lattice.exact_diag import spins_hilbert_space
from tachys.lattice.lattice_database import chain, square
from tachys.lattice.operator.local_estimator import compute_expectation, local_estimator
from tachys.lattice.spins.hamiltonians.heisenberg import heisenberg_square_pbc
from tachys.lattice.spins.hamiltonians.ising_transverse_field import (
    ising_transverse_field_hamiltonian,
)
from tachys.lattice.spins.spin_state import SpinState, init_config_fixed_magn
from tachys.montecarlo import sample
from tachys.wavefunction import WaveFunction

L = 2
N = L * L
N_mc = 8

QS = [0.0, 0.25, 0.7, 1.0]


def _pack(state):
    """Basis index of every configuration, as exact_diag's `pack` callables do."""
    bits = (np.asarray(state.spins) + 1) // 2
    return (bits * 2 ** np.arange(state.spins.shape[-1])).sum(axis=-1).astype(int)


@pytest.fixture(scope="module")
def setup():
    """Heisenberg 2x2 PBC with a fixed RBM, plus the fully enumerated basis."""
    H = heisenberg_square_pbc(L, J=1.0)
    lattice = square(shape=(L, L))

    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    model = SpinRBM(hidden_units=N, dtype=jnp.float64)
    dummy = SpinState(spins=jnp.ones((1, N), dtype=jnp.float64), lattice=lattice)
    params = model.init(k2, dummy)
    # Draw O(1) parameters rather than relying on SpinRBM's default init: a
    # near-uniform |psi| makes every blur weight ~1, which is far too flat to
    # distinguish a correct weight formula from a wrong one. With these,
    # |psi|^2 spans ~3 orders of magnitude and the weights ~4.
    leaves, treedef = jax.tree.flatten(params)
    keys = jax.random.split(k3, len(leaves))
    params = jax.tree.unflatten(treedef, [
        0.4 * jax.random.normal(k, x.shape, dtype=x.dtype) for k, x in zip(keys, leaves)
    ])
    wf = WaveFunction(params=params, apply_fn=model.apply)

    spins = init_config_fixed_magn(k1, N, sz=0, N_mc=N_mc)
    state = SpinState(spins=spins, lattice=lattice)

    full = SpinState(spins=jnp.asarray(spins_hilbert_space(N), dtype=jnp.float64),
                     lattice=lattice)

    return H, lattice, wf, state, full


@pytest.fixture(scope="module")
def exact(setup):
    """p, the connection-count matrix A, N_conn and E_L on the full basis."""
    H, _, wf, _, full = setup
    dim = full.spins.shape[0]

    log_amps = wf.apply_fn(wf.params, full)
    p = np.asarray(jnp.exp(2 * jnp.real(log_amps)))
    p = p / p.sum()

    # A[x, y] = number of off-diagonal terms connecting x -> y.
    _, offdiag = _split_result(H(full))
    cols = np.asarray(_pack(offdiag.connected_states))          # (n_terms, dim)
    active = np.asarray(offdiag.mask).astype(bool)              # (n_terms, dim)
    rows = np.broadcast_to(_pack(full)[None], cols.shape)
    A = np.zeros((dim, dim))
    np.add.at(A, (rows[active], cols[active]), 1.0)

    E_L = np.asarray(local_estimator(H, full, wf, log_amps, optimize_mask=False))
    return p, A, A.sum(1), E_L, log_amps


def _p_blur(p, A, n_conn, q):
    """The blurred density, built independently of the implementation."""
    T = np.divide(A, n_conn[:, None], out=np.zeros_like(A), where=n_conn[:, None] > 0)
    out = (1 - q) * p + q * (p @ T)
    # A walker on a configuration with no active connection can never leave, so
    # it contributes its full weight to the "stayed" term rather than (1-q) of it.
    out[n_conn == 0] += q * p[n_conn == 0]
    return out


def test_connection_multiset_is_symmetric(exact):
    """The weight formula sums over terms active at y in place of arrivals at y;
    that substitution is only valid when #{terms y->x} == #{terms x->y}."""
    _, A, _, _, _ = exact
    assert np.array_equal(A, A.T)


def test_basis_covers_zero_and_nonzero_connectivity(exact):
    """The full-basis fixtures must exercise the N_conn == 0 branch (the two
    fully polarized states of a Heisenberg model), or the guard goes untested."""
    _, _, n_conn, _, _ = exact
    assert n_conn.min() == 0
    assert n_conn.max() > 0


@pytest.mark.parametrize("q", QS)
def test_weight_formula_exact(setup, exact, q):
    """w(y) * p_blur(y) / p(y) must be constant over the whole basis.

    Constant rather than 1 because compute_expectation_blurred normalizes w by
    its mean over the batch, and this "batch" is the uniformly enumerated basis.
    """
    H, _, wf, _, full = setup
    p, A, n_conn, _, _ = exact

    _, w, *_ = compute_expectation_blurred(H, wf, full, jnp.asarray(q, dtype=float))
    ratio = np.asarray(w) * _p_blur(p, A, n_conn, q) / p

    assert np.allclose(ratio, ratio[0], rtol=1e-11, atol=0.0)


@pytest.mark.parametrize("q", QS)
def test_raw_weight_is_exactly_the_density_ratio(setup, exact, q):
    """Unnormalized: w_raw == p / p_blur on the nose, no free constant."""
    H, _, wf, _, full = setup
    p, A, n_conn, _, log_amps = exact

    _, w_raw = blurred_local_estimator(H, full, wf, log_amps, q)
    assert np.allclose(np.asarray(w_raw), p / _p_blur(p, A, n_conn, q), rtol=1e-11)


@pytest.mark.parametrize("q", QS)
def test_estimator_is_unbiased_exact(setup, exact, q):
    """sum_y p_blur(y) w(y) E_L(y) == <H>, with no Monte Carlo error at all."""
    H, _, wf, _, full = setup
    p, A, n_conn, E_L, log_amps = exact

    _, w_raw = blurred_local_estimator(H, full, wf, log_amps, q)
    blurred_mean = float(np.sum(_p_blur(p, A, n_conn, q) * np.asarray(w_raw) * E_L).real)

    assert blurred_mean == pytest.approx(float(np.sum(p * E_L).real), rel=1e-10)


def test_q_zero_matches_standard(setup):
    """q=0 must reduce exactly to the unweighted path: same configurations, unit
    weights, bit-identical local energies."""
    H, _, wf, state, _ = setup
    log_amps = wf.apply_fn(wf.params, state)
    keys = jax.random.split(jax.random.key(3), N_mc)

    eval_state, E_L, weights, e_mean, e2_mean, _ = BlurredEstimator(q=0.0)(
        keys, H, wf, state, log_amps
    )
    E_ref, e_ref, e2_ref = compute_expectation(H, wf, state, log_amps)

    assert jnp.array_equal(eval_state.spins, state.spins)
    assert jnp.allclose(weights, 1.0, atol=1e-14)
    assert jnp.allclose(E_L, E_ref, atol=1e-12)
    assert e_mean == pytest.approx(complex(e_ref), rel=1e-12)
    assert e2_mean == pytest.approx(complex(e2_ref), rel=1e-12)


def test_weights_normalized_to_mean_one(setup):
    """E[w] == 1 under p_blur, so the self-normalised weights average to 1.

    The optimizer no longer needs this (it renormalises internally), but the
    weighted moments compute_expectation_blurred returns do: they are ratio
    estimators, and mean_w is reported as a Monte Carlo diagnostic.
    """
    H, _, wf, state, _ = setup
    log_amps = wf.apply_fn(wf.params, state)
    keys = jax.random.split(jax.random.key(4), N_mc)

    _, _, weights, _, _, metrics = BlurredEstimator(q=0.4)(keys, H, wf, state, log_amps)

    assert float(jnp.mean(weights)) == pytest.approx(1.0, abs=1e-12)
    assert 0.0 < metrics["ess"] <= 1.0


def test_purely_diagonal_operator_rejected(setup):
    """Blurring needs off-diagonal connectivity; a diagonal operator has none."""
    from functools import reduce
    from tachys.lattice.spins.spin_operators import Sz

    _, lattice, wf, state, _ = setup
    diag_only = reduce(lambda a, b: a + b, (Sz(site=i) for i in range(N)))
    log_amps = wf.apply_fn(wf.params, state)

    with pytest.raises(ValueError, match="off-diagonal"):
        blurred_local_estimator(diag_only, state, wf, log_amps, 0.3)


@pytest.mark.parametrize("q", [0.0, 0.3, 1.0])
def test_blur_proposal(setup, exact, q):
    """Every blurred walker either stayed put or moved to a configuration
    connected to its origin, and the moved fraction is q."""
    H, lattice, wf, _, _ = setup
    _, A, n_conn, _, _ = exact

    n_walkers = 4096
    spins = init_config_fixed_magn(jax.random.key(7), N, sz=0, N_mc=n_walkers)
    state = SpinState(spins=spins, lattice=lattice)
    keys = jax.random.split(jax.random.key(8), n_walkers)

    blurred = blur_states(keys, H, state, jnp.asarray(q, dtype=float))

    src, dst = _pack(state), _pack(blurred)
    stayed = src == dst
    # Sz=0 configurations of a 2x2 Heisenberg model always have a valid exchange.
    assert np.all(n_conn[src] > 0)
    assert np.all(A[src[~stayed], dst[~stayed]] > 0)

    moved = float((~stayed).mean())
    tol = 4.0 * np.sqrt(max(q * (1 - q), 1e-12) / n_walkers)
    # A move can land back on the origin only if a term connects a state to
    # itself, which an exchange move never does -- so `moved` estimates q directly.
    assert moved == pytest.approx(q, abs=tol + 1e-12)


@pytest.mark.parametrize("seed", [0, 1])
def test_reweighted_energy_matches_exact_mc(setup, exact, seed):
    """End-to-end: sample -> blur -> reweight recovers <H> within error bars.

    Restricted to the Sz=0 sector, which is what BondExchange explores.
    """
    H, lattice, wf, _, full = setup
    p, _, _, E_L, _ = exact

    sector = np.asarray(full.spins).sum(1) == 0
    p_sec = p[sector] / p[sector].sum()
    e_exact = float(np.sum(p_sec * E_L[sector]).real)

    n_walkers = 8192
    key = jax.random.key(seed)
    key, k_init, k_mc, k_blur = jax.random.split(key, 4)

    state = SpinState(spins=init_config_fixed_magn(k_init, N, sz=0, N_mc=n_walkers),
                      lattice=lattice)
    action = BondExchange.create(lattice, max_dist=1)
    state, log_amps, _ = sample(20, state, action, jax.random.split(k_mc, n_walkers), wf)

    _, E_blur, weights, e_mean, _, metrics = BlurredEstimator(q=0.3)(
        jax.random.split(k_blur, n_walkers), H, wf, state, log_amps
    )

    # Standard error of the self-normalized weighted mean.
    w, e = np.asarray(weights), np.asarray(E_blur).real
    stderr = float(np.std(w * (e - e_mean.real)) / np.sqrt(n_walkers))

    assert metrics["ess"] > 0.5
    assert abs(float(e_mean.real) - e_exact) < 5.0 * stderr + 1e-9


# ─── A second connectivity: TFIM ──────────────────────────────────────────────
#
# The Heisenberg fixtures above have configuration-dependent connectivity that
# conserves Sz, and two configurations with N_conn == 0. TFIM is the opposite on
# every count: sigma_x is active on every site of every configuration, so N_conn
# is the constant Ns and no configuration is a fixed point, and single-site flips
# do not conserve Sz. Running the same exact check on it covers the other regime.

TFIM_L = 8


@pytest.fixture(scope="module")
def tfim():
    lat = chain(TFIM_L, pbc=True)
    H = ising_transverse_field_hamiltonian(lat, nn=[((1, 0), 1.0)], h=1.0)

    model = SpinRBM(hidden_units=TFIM_L, dtype=jnp.float64)
    full = SpinState(spins=jnp.asarray(spins_hilbert_space(TFIM_L), dtype=jnp.float64),
                     lattice=lat)
    params = model.init(jax.random.key(0), full)
    leaves, treedef = jax.tree.flatten(params)
    keys = jax.random.split(jax.random.key(11), len(leaves))
    params = jax.tree.unflatten(treedef, [
        0.4 * jax.random.normal(k, x.shape, dtype=x.dtype) for k, x in zip(keys, leaves)
    ])
    wf = WaveFunction(params=params, apply_fn=model.apply)

    dim = full.spins.shape[0]
    log_amps = wf.apply_fn(wf.params, full)
    p = np.asarray(jnp.exp(2 * jnp.real(log_amps)))
    p = p / p.sum()

    def pack(st):
        bits = (np.asarray(st.spins) + 1) // 2
        return (bits * 2 ** np.arange(st.spins.shape[-1])).sum(axis=-1).astype(int)

    _, offdiag = _split_result(H(full))
    cols = np.asarray(pack(offdiag.connected_states))
    active = np.asarray(offdiag.mask).astype(bool)
    rows = np.broadcast_to(pack(full)[None], cols.shape)
    A = np.zeros((dim, dim))
    np.add.at(A, (rows[active], cols[active]), 1.0)

    return H, wf, full, p, A, A.sum(1), log_amps


def test_tfim_connectivity_is_uniform_and_symmetric(tfim):
    """Every configuration has exactly Ns active single-site flips."""
    _, _, _, _, A, n_conn, _ = tfim
    assert np.array_equal(A, A.T)
    assert np.all(n_conn == TFIM_L)


@pytest.mark.parametrize("q", QS)
def test_tfim_raw_weight_is_exactly_the_density_ratio(tfim, q):
    """w_raw == p / p_blur for a constant-N_conn, Sz-non-conserving operator."""
    H, wf, full, p, A, n_conn, log_amps = tfim

    _, w_raw = blurred_local_estimator(H, full, wf, log_amps, q)
    assert np.allclose(np.asarray(w_raw), p / _p_blur(p, A, n_conn, q), rtol=1e-11)


@pytest.mark.parametrize("q", QS)
def test_tfim_estimator_is_unbiased_exact(tfim, q):
    """sum_y p_blur(y) w(y) E_L(y) == <H>, no Monte Carlo error."""
    H, wf, full, p, A, n_conn, log_amps = tfim

    E_L = np.asarray(local_estimator(H, full, wf, log_amps, optimize_mask=False))
    _, w_raw = blurred_local_estimator(H, full, wf, log_amps, q)
    blurred = float(np.sum(_p_blur(p, A, n_conn, q) * np.asarray(w_raw) * E_L).real)

    assert blurred == pytest.approx(float(np.sum(p * E_L).real), rel=1e-10)


# ─── The train() seam ─────────────────────────────────────────────────────────

def test_train_with_q0_estimator_matches_default_path():
    """`train(estimator=BlurredEstimator(q=0))` must reproduce `train()` exactly.

    This pins the whole seam end to end: the estimator branch, the mean-1 unit
    weights reaching `_call_reweighted` instead of `_call`, and -- crucially --
    the RNG derivation. `train` folds the estimator's keys out of the step's
    sampling key rather than splitting `key` again, precisely so that passing a
    no-op estimator does not shift the Markov trajectory. Split `key` instead and
    the two runs wander apart after step 0 and this test fails.
    """
    from tachys.ground_state_training import train
    from tachys.lattice.spins.hamiltonians.ising_transverse_field import (
        ising_transverse_field_hamiltonian,
    )
    from tachys.lattice.spins.spin_action import SpinFlip
    from tachys.optimizer import SR

    L_t, n_walkers, steps = 6, 32, 5
    lat = chain(L_t, pbc=True)
    H = ising_transverse_field_hamiltonian(lat, nn=[((1, 0), 1.0)], h=1.0)
    model = SpinRBM(hidden_units=L_t, dtype=jnp.float64)

    def go(est):
        key = jax.random.key(0)
        key, ki, kp = jax.random.split(key, 3)
        spins = jnp.where(jax.random.bernoulli(ki, 0.5, (n_walkers, L_t)), 1.0, -1.0)
        st = SpinState(spins=spins, lattice=lat)
        wf = WaveFunction(params=model.init(kp, st), apply_fn=model.apply)
        *_, hist = train(key, H, st, wf, SR(diag_shift=1e-3, mode="real"), SpinFlip(),
                         steps, lambda t: 0.01, n_walkers, estimator=est)
        return np.array(hist["energy"])

    assert np.allclose(go(None), go(BlurredEstimator(q=0.0)), rtol=0, atol=1e-11)
