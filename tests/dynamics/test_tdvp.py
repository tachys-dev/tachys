"""The real-time TDVP velocity.

Two independent anchors, neither of which reuses the implementation's algebra:

1. ``test_velocity_matches_the_dense_tdvp_equation`` builds ``S`` and ``F``
   explicitly from ``jax.jacobian`` over the *same* samples and solves
   ``S thetadot = Im F`` with a plain pseudo-inverse. Any error in the ``1j``,
   the conjugation, the dropped factor of 2, the block layout or the centering
   shows up as a mismatch.
2. ``test_exactly_representable_phase_evolution`` uses an ansatz for which the
   answer is known in closed form and is independent of the sample set:
   ``psi_theta(x) = exp(-i theta V(x))`` with ``H|x> = V(x)|x>`` solves the
   Schrodinger equation exactly for ``theta(t) = t``, so TDVP must return
   ``dtheta/dt = 1`` with no Monte Carlo error whatsoever. This pins the sign of
   time -- a flipped sign gives -1, which the dense check above would miss if it
   shared the error.
"""
from typing import Any

import numpy as np
import pytest

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from tachys.dynamics import TDVP
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
    """log psi(x) = -i * theta * V(x) with V(x) = -sum_<ij> s_i s_j.

    Exactly the phase accumulated under the classical Ising Hamiltonian
    H = -sum_<ij> sigma^z_i sigma^z_j, so theta(t) = t is an exact solution.
    """
    src: Any
    dst: Any

    @nn.compact
    def __call__(self, state):
        theta = self.param('theta', lambda k: jnp.zeros((1,), jnp.float64))
        s = jnp.atleast_2d(state.spins)
        V = -jnp.sum(s[:, jnp.asarray(self.src)] * s[:, jnp.asarray(self.dst)], axis=-1)
        return -1j * theta[0] * V.astype(jnp.float64)


def _sampled_batch(model, H, seed=2, hidden=2):
    lattice = square(shape=(L, L))
    spins = init_config_fixed_magn(jax.random.key(1), N, sz=0, N_mc=N_mc)
    state = SpinState(spins=spins, lattice=lattice)
    params = model.init(jax.random.key(0), state)
    wf = WaveFunction(params=params, apply_fn=model.apply)
    state, log_amps, _ = sample(1, state, SpinFlip(), jax.random.split(jax.random.key(seed), N_mc), wf)
    E_L, e_mean, e2_mean = compute_expectation(H, wf, state, log_amps)
    return state, wf, E_L, e_mean, e2_mean


def _dense_S_and_F(wf, state, E_L):
    """S = Re<DeltaO* DeltaO> and F = <DeltaO* DeltaE_L>, built from scratch."""
    flat_params, unravel = ravel_pytree(wf.params)

    def log_psi_flat(theta_flat):
        return wf.apply_fn(unravel(theta_flat), state)

    O = jax.jacfwd(log_psi_flat, holomorphic=False)(flat_params)   # (N_mc, P), complex
    O = np.asarray(O)
    dO = O - O.mean(axis=0, keepdims=True)
    dE = np.asarray(E_L) - np.asarray(E_L).mean()
    S = np.real(dO.conj().T @ dO) / len(dE)
    F = (dO.conj().T @ dE) / len(dE)
    return S, F


def test_velocity_matches_the_dense_tdvp_equation():
    """S thetadot = Im F, with S and F assembled independently of the NTK path."""
    H = ising_transverse_field_square_pbc(L, J=1.0, h=1.0)
    model = SpinRBM(hidden_units=2, dtype=jnp.float64, complex=True)
    state, wf, E_L, _, _ = _sampled_batch(model, H)

    tdvp = TDVP(mode="complex", rcond=1e-10)
    dtheta, _ = tdvp(E_L, tdvp.init(wf.params), state, wf)
    got, _ = ravel_pytree(dtheta)

    S, F = _dense_S_and_F(wf, state, E_L)
    expected = np.linalg.pinv(S, rcond=1e-10, hermitian=True) @ np.imag(F)

    assert np.allclose(np.asarray(got), expected, atol=1e-9), \
        f"max |diff| = {np.abs(np.asarray(got) - expected).max():.3e}"
    # and the residual of the equation itself is zero, which also rules out a
    # solution that merely happens to be close in norm
    assert np.allclose(S @ np.asarray(got), np.imag(F), atol=1e-9)


def test_velocity_is_not_the_imaginary_time_direction():
    """Guard against silently reusing the ground-state force: Re F and Im F are
    independent, so the two velocities must differ substantially."""
    H = ising_transverse_field_square_pbc(L, J=1.0, h=1.0)
    model = SpinRBM(hidden_units=2, dtype=jnp.float64, complex=True)
    state, wf, E_L, _, _ = _sampled_batch(model, H)

    tdvp = TDVP(mode="complex", rcond=1e-10)
    dtheta, _ = tdvp(E_L, tdvp.init(wf.params), state, wf)
    got, _ = ravel_pytree(dtheta)

    S, F = _dense_S_and_F(wf, state, E_L)
    imaginary_time = np.linalg.pinv(S, rcond=1e-10, hermitian=True) @ (2 * np.real(F))
    assert not np.allclose(np.asarray(got), imaginary_time, atol=1e-3)


def test_exactly_representable_phase_evolution():
    """psi = exp(-i theta V) under a diagonal H must give dtheta/dt exactly 1.

    Sample-set independent: the residual is zero configuration by configuration,
    so this holds for any batch and pins the sign, the factor of 2 and the
    normalization all at once.
    """
    src, dst = _square_bonds(L)
    H = -4.0 * Sz(src) * Sz(dst)                # = -sum_<ij> sigma^z_i sigma^z_j
    model = DiagonalPhase(src=src, dst=dst)
    state, wf, E_L, _, _ = _sampled_batch(model, H)

    tdvp = TDVP(mode="complex", rcond=1e-10)
    dtheta, _ = tdvp(E_L, tdvp.init(wf.params), state, wf)
    assert float(dtheta['params']['theta'][0]) == pytest.approx(1.0, abs=1e-10)


def test_tdvp_rejects_real_mode():
    with pytest.raises(ValueError, match="requires mode='complex'"):
        TDVP(mode="real")


def test_tdvp_state_is_empty_and_stateless():
    tdvp = TDVP(mode="complex")
    opt_state = tdvp.init({"a": jnp.zeros(3)})
    assert jax.tree.leaves(opt_state) == []


def test_diag_shift_defaults_to_zero():
    """With the spectral truncation doing the regularizing, a Tikhonov shift is
    not needed and would bias the retained directions."""
    assert np.all(np.asarray(TDVP(mode="complex").diag_shift) == 0.0)
