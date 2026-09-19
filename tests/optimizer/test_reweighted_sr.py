"""Tests for the importance-weighted branch of the SR-family optimizers.

`_BaseOptimizer.__call__(..., weights=...)` and the weighted paths in
`_kernels.py` predate any caller; blurred sampling
(tachys.experimental.blurred_sampling) is the first one, and it arrived with no
numerical coverage of that branch at all. These tests pin the four properties it
has to satisfy:

* with unit weights it must reproduce the unweighted update exactly,
* it must reproduce the dense weighted-SR formula, and the exact analytic
  weighted-SR update on a fully enumerated basis,
* the update must not depend on the energy offset, and
* the update must not depend on the overall scale of the weights.

The last one is worth a note. `w` and `c*w` encode the same distribution, so a
correct estimator is scale-free -- but every reduction on this path is
`jnp.mean(w * X)` rather than `sum(wX)/sum(w)`, which is only the weighted mean
when `sum_i w_i == M`. `_call_reweighted` therefore divides by the psum'd mean
weight before anything downstream sees the weights, which is what makes the
scale-invariance below hold -- and it is also what lets `_center_eloc` write the
weighted mean as `mean(w * E_L)`. Which constant `_center_eloc` subtracts does
not actually matter: centring the Jacobian with the weighted mean puts sqrt(w)
in the null space of the centred O^T, so any constant subtracted from E_L gives
the same update (pinned by test_update_invariant_to_energy_shift).
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

from tachys.lattice.ansatz.rbm import SpinRBM
from tachys.lattice.lattice_database import square
from tachys.lattice.operator.local_estimator import local_estimator
from tachys.lattice.spins.hamiltonians.heisenberg import heisenberg_square_pbc
from tachys.lattice.spins.spin_state import SpinState, init_config_fixed_magn
from tachys.optimizer import MARCH, SPRING, SR
from tachys.wavefunction import WaveFunction

L = 4
N = L * L
N_mc = 16
DIAG_SHIFT = 1e-3


def _setup(mode):
    H = heisenberg_square_pbc(L, J=1.0)
    lattice = square(shape=(L, L))

    model = SpinRBM(hidden_units=2, dtype=jnp.float64, complex=(mode == "complex"))
    spins = init_config_fixed_magn(jax.random.key(1), N, sz=0, N_mc=N_mc)
    state = SpinState(spins=spins, lattice=lattice)
    params = model.init(jax.random.key(0), state)
    # Draw O(1) parameters rather than relying on SpinRBM's default init: a
    # near-uniform psi gives a degenerate Jacobian and near-constant local
    # energies, which would not exercise the weighting algebra at all.
    leaves, treedef = jax.tree.flatten(params)
    keys = jax.random.split(jax.random.key(2), len(leaves))
    params = jax.tree.unflatten(treedef, [
        0.4 * jax.random.normal(k, x.shape, dtype=x.dtype) for k, x in zip(keys, leaves)
    ])
    wf = WaveFunction(params=params, apply_fn=model.apply)

    log_amps = wf.apply_fn(wf.params, state)
    E_L = local_estimator(H, state, wf, log_amps, optimize_mask=False)
    return state, wf, E_L


def _flat(updates):
    return ravel_pytree(updates)[0]


@pytest.mark.parametrize("mode", ["real", "complex"])
@pytest.mark.parametrize(
    "make_optimizer",
    [
        partial(SR, diag_shift=DIAG_SHIFT),
        partial(SPRING, diag_shift=DIAG_SHIFT, mu=0.9),
        partial(MARCH, diag_shift=DIAG_SHIFT, mu=0.95, beta=0.995),
    ],
    ids=["SR", "SPRING", "MARCH"],
)
def test_unit_weights_match_unweighted(make_optimizer, mode):
    """weights = 1 must reproduce the weights=None update.

    This exercises every weighted insertion point at once: the sqrt(w) scaling of
    eps, center_ntk, compute_ntk, center_sr_solution, _jvp_correction and the
    weighted mean in _center_eloc.
    """
    state, wf, E_L = _setup(mode)
    optimizer = make_optimizer(mode=mode)
    opt_state = optimizer.init(wf.params)

    plain, _ = optimizer(E_L, opt_state, state, wf)
    weighted, _ = optimizer(E_L, opt_state, state, wf, weights=jnp.ones(N_mc))

    assert jnp.allclose(_flat(plain), _flat(weighted), rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("mode", ["real", "complex"])
@pytest.mark.parametrize(
    "make_optimizer",
    [
        partial(SR, diag_shift=DIAG_SHIFT),
        partial(SPRING, diag_shift=DIAG_SHIFT, mu=0.9),
        partial(MARCH, diag_shift=DIAG_SHIFT, mu=0.95, beta=0.995),
    ],
    ids=["SR", "SPRING", "MARCH"],
)
def test_update_invariant_to_energy_shift(make_optimizer, mode):
    """E_L -> E_L + c must leave the weighted update unchanged.

    The natural gradient does not depend on where the energy zero sits. Pinning
    it on the weighted branch checks that no weight factor sneaks into `eloc`
    asymmetrically with the centring that produced it.
    """
    state, wf, E_L = _setup(mode)
    optimizer = make_optimizer(mode=mode)
    opt_state = optimizer.init(wf.params)

    w = jnp.asarray(np.linspace(0.2, 1.8, N_mc))
    w = w / jnp.mean(w)

    base, _ = optimizer(E_L, opt_state, state, wf, weights=w)
    shifted, _ = optimizer(E_L + 7.5, opt_state, state, wf, weights=w)

    scale = float(jnp.max(jnp.abs(_flat(base))))
    assert jnp.max(jnp.abs(_flat(base) - _flat(shifted))) < 1e-8 * scale


def _dense_weighted_sr(state, wf, E_L, weights, diag_shift):
    """Reference weighted SR (mode="real") built from an explicit Jacobian.

    Mirrors jaxvmcf's kernel_SR_with_weights: weighted-mean-centred O, sqrt(w) on
    both O and de, and 2 * conj(de) as the right-hand side.
    """
    w = np.asarray(weights)
    n = N_mc

    def _f(params, single):
        return jnp.squeeze(wf.apply_fn(params, single)).real

    per_sample = jax.tree.map(lambda x: x[:, None], state)   # leaves (N_mc, 1, ...)
    jac = jax.vmap(jax.grad(_f), in_axes=(None, 0))(wf.params, per_sample)
    O = np.asarray(jax.vmap(lambda t: ravel_pytree(t)[0])(jac))   # (N_mc, N_params)

    de = np.asarray(E_L) - np.sum(w * np.asarray(E_L)) / n
    O = O - (w[:, None] * O).mean(0, keepdims=True)
    O = O * np.sqrt(w)[:, None] / np.sqrt(n)
    dv = (2.0 * np.conj(de * np.sqrt(w)) / np.sqrt(n)).real

    ntk = O @ O.T
    a = np.linalg.solve(ntk + diag_shift * np.eye(n), dv)
    return O.T @ a


@pytest.mark.parametrize("mode", ["real"])
def test_weighted_sr_matches_dense_reference(mode):
    """SR's weighted update equals the dense weighted-SR formula."""
    state, wf, E_L = _setup(mode)
    optimizer = SR(diag_shift=DIAG_SHIFT, mode=mode)
    opt_state = optimizer.init(wf.params)

    w = jnp.asarray(np.linspace(0.3, 1.7, N_mc))
    w = w / jnp.mean(w)

    updates, _ = optimizer(E_L, opt_state, state, wf, weights=w)
    reference = _dense_weighted_sr(state, wf, E_L, w, DIAG_SHIFT)

    assert np.allclose(np.asarray(_flat(updates)), reference, rtol=1e-6, atol=1e-8)


# ─── Exact ground truth ───────────────────────────────────────────────────────

def _exact_setup(N=6, scale=0.3):
    """Whole enumerated basis as the sample set, with w_i = M * p_i.

    Then (1/M) sum_i w_i f_i == sum_i p_i f_i exactly and mean(w) == 1, so the
    weighted SR update on this set *is* the exact SR update -- no Monte Carlo
    error anywhere, and the exact answer is computable in closed form.
    """
    from tachys.lattice.exact_diag import spins_hilbert_space
    from tachys.lattice.lattice_database import chain
    from tachys.lattice.spins.hamiltonians.ising_transverse_field import (
        ising_transverse_field_hamiltonian,
    )

    lat = chain(N, pbc=True)
    H = ising_transverse_field_hamiltonian(lat, nn=[((1, 0), 1.0)], h=1.0)
    full = SpinState(spins=jnp.asarray(spins_hilbert_space(N), dtype=jnp.float64),
                     lattice=lat)
    M = full.spins.shape[0]

    model = SpinRBM(hidden_units=N, dtype=jnp.float64)
    params = model.init(jax.random.key(0), full)
    leaves, treedef = jax.tree.flatten(params)
    ks = jax.random.split(jax.random.key(7), len(leaves))
    params = jax.tree.unflatten(treedef, [
        scale * jax.random.normal(k, x.shape, dtype=x.dtype) for k, x in zip(ks, leaves)
    ])
    wf = WaveFunction(params=params, apply_fn=model.apply)

    log_amps = wf.apply_fn(wf.params, full)
    p = np.asarray(jnp.exp(2 * jnp.real(log_amps)))
    p = p / p.sum()
    E_L = np.asarray(local_estimator(H, full, wf, log_amps, optimize_mask=False)).real

    def _f(prm, single):
        return jnp.squeeze(wf.apply_fn(prm, single)).real

    jac = jax.vmap(jax.grad(_f), in_axes=(None, 0))(
        wf.params, jax.tree.map(lambda x: x[:, None], full))
    O = np.asarray(jax.vmap(lambda t: ravel_pytree(t)[0])(jac))     # (M, P)

    return full, wf, jnp.asarray(E_L), jnp.asarray(M * p), p, E_L, O


def _analytic_sr(p, E_L, O, center, lam):
    """dtheta = (S + lam I)^-1 g, the textbook weighted SR update.

    tachys solves A^T (A A^T + lam I)^-1 eps; the identity
    A^T(AA^T + lam I)^-1 == (A^T A + lam I)^-1 A^T makes that the same object.
    """
    o_bar = (p[:, None] * O).sum(0)
    Oc = O - o_bar
    S = (p[:, None] * Oc).T @ Oc
    g = 2.0 * (p * (E_L - center)) @ Oc
    return np.linalg.solve(S + lam * np.eye(O.shape[1]), g)


def test_weighted_sr_matches_exact_ground_truth():
    """tachys' weighted SR reproduces the analytic weighted SR update.

    Also shows the E_L centring constant is irrelevant: the weighted mean and the
    plain mean differ substantially here, yet give the same update.
    """
    full, wf, E_L_j, w, p, E_L, O = _exact_setup()

    e_weighted = float((p * E_L).sum())
    e_plain = float(E_L.mean())
    assert abs(e_plain - e_weighted) > 0.5, "centring constants must differ for this to bite"

    d_weighted = _analytic_sr(p, E_L, O, e_weighted, DIAG_SHIFT)
    d_plain = _analytic_sr(p, E_L, O, e_plain, DIAG_SHIFT)
    d_none = _analytic_sr(p, E_L, O, 0.0, DIAG_SHIFT)

    scale = np.abs(d_weighted).max()
    assert np.abs(d_weighted - d_plain).max() < 1e-9 * scale
    assert np.abs(d_weighted - d_none).max() < 1e-9 * scale

    opt = SR(diag_shift=DIAG_SHIFT, mode="real")
    upd, _ = opt(E_L_j, opt.init(wf.params), full, wf, weights=w)
    assert np.abs(np.asarray(_flat(upd)) - d_weighted).max() < 1e-9 * scale


@pytest.mark.parametrize("c", [1.0, 2.0, 0.5, 137.0, 1e-3])
def test_weighted_sr_is_invariant_to_weight_scale(c):
    """w and c*w encode the same distribution, so both must give the same update.

    The reductions downstream are `jnp.mean(w * X)`, which is the weighted mean
    only for `sum_i w_i == M`; `_call_reweighted` normalizes to that before
    calling `update`, so callers may pass weights at any scale. Without that
    normalization `c = 2` moves the update by >200%.
    """
    full, wf, E_L_j, w, p, E_L, O = _exact_setup()
    exact = _analytic_sr(p, E_L, O, float((p * E_L).sum()), DIAG_SHIFT)

    opt = SR(diag_shift=DIAG_SHIFT, mode="real")
    upd, _ = opt(E_L_j, opt.init(wf.params), full, wf, weights=c * w)
    assert np.abs(np.asarray(_flat(upd)) - exact).max() < 1e-9 * np.abs(exact).max()


@pytest.mark.parametrize("mode", ["real", "complex"])
@pytest.mark.parametrize(
    "make_optimizer",
    [
        partial(SR, diag_shift=DIAG_SHIFT),
        partial(SPRING, diag_shift=DIAG_SHIFT, mu=0.9),
        partial(MARCH, diag_shift=DIAG_SHIFT, mu=0.95, beta=0.995),
    ],
    ids=["SR", "SPRING", "MARCH"],
)
def test_weight_scale_invariance_all_optimizers(make_optimizer, mode):
    """Scale invariance holds for every optimizer, not just SR.

    SPRING and MARCH also feed the weights through `_jvp_correction`, so the
    single normalization in `_call_reweighted` has to cover that path too.
    """
    state, wf, E_L = _setup(mode)
    optimizer = make_optimizer(mode=mode)
    opt_state = optimizer.init(wf.params)

    w = jnp.asarray(np.linspace(0.2, 1.8, N_mc))
    base, _ = optimizer(E_L, opt_state, state, wf, weights=w / jnp.mean(w))
    scaled, _ = optimizer(E_L, opt_state, state, wf, weights=17.0 * w)

    assert jnp.allclose(_flat(base), _flat(scaled), rtol=1e-9, atol=1e-11)
