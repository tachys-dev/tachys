"""The integrated TDVP error R^2 and its per-step rate.

The rate is checked against the definition written out literally --
``Var(H) + thetadot^T S thetadot - 2 Im(F)^T thetadot`` with ``S`` and ``F``
assembled densely from ``jax.jacobian`` -- rather than against the fused form the
implementation actually evaluates, so the two routes are genuinely independent.
Then the three limits that fix the normalization: zero velocity gives ``Var(H)``,
an exactly representable evolution gives zero, and the TDVP solution gives
``ratio == 1``.
"""
from typing import Any

import numpy as np
import pytest

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from tachys.dynamics import TDVP, tdvp_error_rate
from tachys.dynamics.error import TDVPError
from tachys.lattice.ansatz.rbm import SpinRBM
from tachys.lattice.lattice_database import square
from tachys.lattice.operator.local_estimator import compute_expectation
from tachys.lattice.spins.hamiltonians.ising_transverse_field import (
    ising_transverse_field_square_pbc,
)
from tachys.lattice.spins.spin_action import SpinFlip
from tachys.lattice.spins.spin_operators import Sz
from tachys.lattice.spins.spin_state import SpinState, init_config_fixed_magn
from tachys.montecarlo import sample
from tachys.wavefunction import WaveFunction

L = 4
N = L * L
N_mc = 24


def _square_bonds(L):
    bonds = [(x * L + y, x * L + (y + 1) % L) for x in range(L) for y in range(L)]
    bonds += [(x * L + y, ((x + 1) % L) * L + y) for x in range(L) for y in range(L)]
    return zip(*bonds)


class DiagonalPhase(nn.Module):
    """log psi(x) = -i theta V(x): an exact solution of the Schrodinger equation
    for a diagonal H, so its TDVP residual is exactly zero."""
    src: Any
    dst: Any

    @nn.compact
    def __call__(self, state):
        theta = self.param('theta', lambda k: jnp.zeros((1,), jnp.float64))
        s = jnp.atleast_2d(state.spins)
        V = -jnp.sum(s[:, jnp.asarray(self.src)] * s[:, jnp.asarray(self.dst)], axis=-1)
        return -1j * theta[0] * V.astype(jnp.float64)


def _batch(model, H, seed=2):
    lattice = square(shape=(L, L))
    spins = init_config_fixed_magn(jax.random.key(1), N, sz=0, N_mc=N_mc)
    state = SpinState(spins=spins, lattice=lattice)
    wf = WaveFunction(params=model.init(jax.random.key(0), state), apply_fn=model.apply)
    state, log_amps, _ = sample(1, state, SpinFlip(), jax.random.split(jax.random.key(seed), N_mc), wf)
    E_L, e_mean, e2_mean = compute_expectation(H, wf, state, log_amps)
    return state, wf, E_L, e_mean, e2_mean


def _dense_rate(wf, state, E_L, dtheta):
    """delta s^2 / delta t^2, written out term by term from the definition."""
    flat, unravel = ravel_pytree(wf.params)
    O = np.asarray(jax.jacfwd(lambda th: wf.apply_fn(unravel(th), state))(flat))
    dO = O - O.mean(axis=0, keepdims=True)
    dE = np.asarray(E_L) - np.asarray(E_L).mean()
    M = len(dE)
    S = np.real(dO.conj().T @ dO) / M
    F = (dO.conj().T @ dE) / M
    v, _ = ravel_pytree(dtheta)
    v = np.asarray(v)
    var_H = float(np.mean(np.abs(dE) ** 2))
    quad = float(v @ S @ v)
    force = float(2 * v @ np.imag(F))        # 2 Im(F)^T thetadot, F_k = <DeltaO_k* DeltaE_L>
    return var_H + quad - force, var_H, quad, force


def test_rate_matches_the_literal_definition():
    """For an arbitrary (non-TDVP) velocity, so the three terms are all nonzero
    and none of them can cancel an error in another."""
    H = ising_transverse_field_square_pbc(L, J=1.0, h=1.0)
    model = SpinRBM(hidden_units=2, dtype=jnp.float64, complex=True)
    state, wf, E_L, _, _ = _batch(model, H)

    key = jax.random.key(11)
    dtheta = jax.tree.map(
        lambda p, k: 0.3 * jax.random.normal(k, p.shape, p.dtype),
        wf.params, jax.tree.unflatten(jax.tree.structure(wf.params),
                                      list(jax.random.split(key, len(jax.tree.leaves(wf.params))))))

    est = tdvp_error_rate(wf, state, E_L, dtheta, mode="complex")
    rate, var_H, quad, force = _dense_rate(wf, state, E_L, dtheta)

    assert float(est.var_H) == pytest.approx(var_H, rel=1e-10)
    assert float(est.quad) == pytest.approx(quad, rel=1e-10)
    assert float(est.force) == pytest.approx(force, rel=1e-10)
    assert float(est.rate) == pytest.approx(rate, rel=1e-8)
    # the fused and decomposed forms agree (they are algebraically identical)
    assert float(est.decomposed) == pytest.approx(float(est.rate), rel=1e-8)


def test_zero_velocity_gives_the_variance_of_H():
    """A frozen ansatz accumulates error at exactly Var(H) -- the baseline that
    fixes the normalization of the rate."""
    H = ising_transverse_field_square_pbc(L, J=1.0, h=1.0)
    model = SpinRBM(hidden_units=2, dtype=jnp.float64, complex=True)
    state, wf, E_L, e_mean, e2_mean = _batch(model, H)

    zero = jax.tree.map(jnp.zeros_like, wf.params)
    est = tdvp_error_rate(wf, state, E_L, zero, mode="complex")

    var_H = float(jnp.real(e2_mean)) - abs(complex(e_mean)) ** 2
    assert float(est.rate) == pytest.approx(var_H, rel=1e-10)
    assert float(est.quad) == pytest.approx(0.0, abs=1e-14)
    assert float(est.force) == pytest.approx(0.0, abs=1e-14)
    # and Var(H) really is <|E_L|^2> - |<E_L>|^2 as compute_expectation reports it
    assert float(est.var_H) == pytest.approx(var_H, rel=1e-10)


def test_rate_is_zero_for_an_exactly_representable_evolution():
    """psi = exp(-i theta V) under a diagonal H has no variational error at all."""
    src, dst = _square_bonds(L)
    H = -4.0 * Sz(src) * Sz(dst)
    model = DiagonalPhase(src=src, dst=dst)
    state, wf, E_L, _, _ = _batch(model, H)

    tdvp = TDVP(mode="complex", rcond=1e-10)
    dtheta, _ = tdvp(E_L, tdvp.init(wf.params), state, wf)
    est = tdvp_error_rate(wf, state, E_L, dtheta, mode="complex")

    assert float(est.var_H) > 1.0                      # the error is small, not the scale
    assert float(est.rate) == pytest.approx(0.0, abs=1e-18)


def test_ratio_is_one_at_the_tdvp_solution():
    """force / (2 quad) -> 1 iff thetadot solves S thetadot = Im F, so it catches
    exactly the factor-of-2 and sign errors the force vector is prone to."""
    H = ising_transverse_field_square_pbc(L, J=1.0, h=1.0)
    model = SpinRBM(hidden_units=2, dtype=jnp.float64, complex=True)
    state, wf, E_L, _, _ = _batch(model, H)

    tdvp = TDVP(mode="complex", rcond=1e-10)
    dtheta, _ = tdvp(E_L, tdvp.init(wf.params), state, wf)
    est = tdvp_error_rate(wf, state, E_L, dtheta, mode="complex")
    assert float(est.ratio) == pytest.approx(1.0, rel=1e-8)

    # doubling the velocity halves the ratio -- the diagnostic's whole purpose
    doubled = jax.tree.map(lambda x: 2.0 * x, dtheta)
    assert float(tdvp_error_rate(wf, state, E_L, doubled, mode="complex").ratio) \
        == pytest.approx(0.5, rel=1e-8)


def test_rate_is_never_negative_even_for_a_wildly_wrong_velocity():
    """The fused estimator is a mean of squared moduli, so there is nothing to clip."""
    H = ising_transverse_field_square_pbc(L, J=1.0, h=1.0)
    model = SpinRBM(hidden_units=2, dtype=jnp.float64, complex=True)
    state, wf, E_L, _, _ = _batch(model, H)

    tdvp = TDVP(mode="complex", rcond=1e-10)
    dtheta, _ = tdvp(E_L, tdvp.init(wf.params), state, wf)
    for scale in (-100.0, -1.0, 0.0, 1.0, 100.0):
        bad = jax.tree.map(lambda x: scale * x, dtheta)
        assert float(tdvp_error_rate(wf, state, E_L, bad, mode="complex").rate) >= 0.0


class _Est:
    """Stand-in for a TDVPErrorEstimate with a prescribed rate."""
    def __init__(self, rate):
        self.rate = jnp.asarray(rate)
        self.var_H = jnp.asarray(1.0)
        self.quad = jnp.asarray(0.0)
        self.force = jnp.asarray(0.0)
        self.ratio = jnp.asarray(1.0)
        self.decomposed = self.rate


def test_accumulator_is_a_riemann_sum_of_the_definition():
    """R^2(t) = (1/sqrt(N)) sum sqrt(delta s^2), with sqrt(delta s^2) = dt sqrt(rate)."""
    Ns, dt = 9, 0.1
    acc = TDVPError(rule="rect")

    # constant rate r over 4 measured steps: R^2 = (1/sqrt(N)) * 3*dt*sqrt(r)
    # (three intervals between four measurements; nothing is added at the first)
    r = 4.0
    for step in range(4):
        acc.accumulate(step, step * dt, Ns, _Est(r))
    assert acc.R2 == pytest.approx(3 * dt * np.sqrt(r) / np.sqrt(Ns))

    # nothing accumulated on a single measurement
    assert TDVPError().accumulate(0, 0.0, Ns, _Est(r)) is not None
    assert TDVPError().R2 == 0.0


def test_accumulator_uses_elapsed_time_not_a_fixed_stride():
    """Measuring every n steps is the rectangle rule of width n*dt, and a changed
    stride or a skipped measurement must still integrate the right interval."""
    Ns = 1
    acc = TDVPError(rule="rect")
    acc.accumulate(0, 0.0, Ns, _Est(1.0))     # sqrt(rate) = 1
    acc.accumulate(5, 0.5, Ns, _Est(4.0))     # interval [0.0, 0.5), left value 1
    assert acc.R2 == pytest.approx(0.5 * 1.0)
    acc.accumulate(7, 0.7, Ns, _Est(9.0))     # interval [0.5, 0.7), left value 2
    assert acc.R2 == pytest.approx(0.5 * 1.0 + 0.2 * 2.0)


def test_accumulator_trapezoid_rule():
    Ns = 1
    acc = TDVPError(rule="trapezoid")
    acc.accumulate(0, 0.0, Ns, _Est(1.0))     # sqrt = 1
    acc.accumulate(1, 0.5, Ns, _Est(9.0))     # sqrt = 3
    assert acc.R2 == pytest.approx(0.5 * 0.5 * (1.0 + 3.0))


def test_accumulator_validates_its_arguments():
    with pytest.raises(ValueError, match="rule must be"):
        TDVPError(rule="simpson")
