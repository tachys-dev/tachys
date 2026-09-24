"""Operators return rows of their matrix.

``apply`` on a configuration x returns the configurations x' with
<x|O|x'> != 0, together with those matrix elements. That is what the local
estimator sum_x' <x|O|x'> psi(x') / psi(x) needs for its average over |psi|^2
to be <psi|O|psi>, for any operator, Hermitian or not, and what
``build_sparse_hamiltonian`` needs to return the matrix of O rather than its
transpose.

Each test enumerates every configuration of a two-site system, so the averages
are exact, and compares against dense matrices built independently with numpy
(Kronecker products; Jordan-Wigner strings for the fermions). The amplitudes of
the lookup-table wavefunction are complex, so that O and its transpose give
different results whenever O has complex or non-symmetric matrix elements.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

from tachys.dynamics import TDVP
from tachys.lattice.exact_diag import build_sparse_hamiltonian
from tachys.lattice.fermions.fermion_operators import (
    Cdn, Cdn_dag, Cup, Cup_dag, HoppingDown, HoppingUp, Ndn, Nup,
)
from tachys.lattice.fermions.fermion_state import FermionState
from tachys.lattice.lattice_database import chain
from tachys.lattice.operator.local_estimator import compute_expectation, local_estimator
from tachys.lattice.spins.spin_operators import Sminus, Splus, Sx, Sy, Sz, XYExchange
from tachys.lattice.spins.spin_state import SpinState
from tachys.optimizer import SR
from tachys.wavefunction import WaveFunction

LAT = chain(2, pbc=False)
RNG = np.random.default_rng(0)


def _lookup_wavefunction(amps, index):
    """Wavefunction whose amplitude on configuration x is amps[index(x)]."""
    table = jnp.log(jnp.asarray(amps, dtype=jnp.complex128))

    def apply_fn(params, state):
        return params[index(state)]

    return WaveFunction(params=table, apply_fn=apply_fn)


def _exact_average(op, basis_state, amps, index):
    """sum_x |psi(x)|^2 O_L(x), over every configuration x."""
    wf = _lookup_wavefunction(amps, index)
    log_amps = wf.apply_fn(wf.params, basis_state)
    O_L = np.asarray(local_estimator(op, basis_state, wf, log_amps, optimize_mask=False))
    weights = np.abs(amps) ** 2
    return np.sum(weights * O_L) / np.sum(weights)


# ---------------------------------------------------------------- spins -----
# Basis |s0 s1>, index 2*b0 + b1 with b = 0 for up and 1 for down, so that the
# dense matrices below are Kronecker products with site 0 as the left factor.

SPIN_BASIS = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=np.int8)
SPIN_STATE = SpinState(spins=jnp.asarray(SPIN_BASIS), lattice=LAT)
SPIN_AMPS = RNG.normal(size=4) + 1j * RNG.normal(size=4)


def _spin_index(state):
    b = (1 - jnp.atleast_2d(state.spins)) // 2
    return b[..., 0] * 2 + b[..., 1]


_S = {
    "z": np.diag([0.5, -0.5]),
    "x": np.array([[0, 0.5], [0.5, 0]]),
    "y": np.array([[0, -0.5j], [0.5j, 0]]),
    "+": np.array([[0, 1.0], [0, 0]]),
    "-": np.array([[0, 0], [1.0, 0]]),
}


def _on(site, name):
    return np.kron(_S[name], np.eye(2)) if site == 0 else np.kron(np.eye(2), _S[name])


SPIN_CASES = {
    "Sz(0)":                   (lambda: Sz(0), _on(0, "z")),
    "Sx(1)":                   (lambda: Sx(1), _on(1, "x")),
    "Sy(0)":                   (lambda: Sy(0), _on(0, "y")),
    "Splus(0)":                (lambda: Splus(0), _on(0, "+")),
    "Sminus(1)":               (lambda: Sminus(1), _on(1, "-")),
    "Splus(0) * Sminus(1)":    (lambda: Splus(0) * Sminus(1), _on(0, "+") @ _on(1, "-")),
    "Sy(0) * Splus(1)":        (lambda: Sy(0) * Splus(1), _on(0, "y") @ _on(1, "+")),
    "Sz(0) * Splus(0)":        (lambda: Sz(0) * Splus(0), _on(0, "z") @ _on(0, "+")),
    "(0.3+0.2j) Sminus(0) Sy(1)": (lambda: (0.3 + 0.2j) * Sminus(0) * Sy(1),
                                   (0.3 + 0.2j) * _on(0, "-") @ _on(1, "y")),
    "XYExchange(0, 1)":        (lambda: XYExchange(0, 1),
                                _on(0, "+") @ _on(1, "-") + _on(0, "-") @ _on(1, "+")),
}


@pytest.mark.parametrize("name", list(SPIN_CASES))
def test_spin_expectation_is_psi_O_psi(name):
    make_op, dense = SPIN_CASES[name]
    psi = SPIN_AMPS / np.linalg.norm(SPIN_AMPS)
    expected = np.vdot(psi, dense @ psi)
    got = _exact_average(make_op(), SPIN_STATE, SPIN_AMPS, _spin_index)
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_spin_sparse_matrix_is_O_not_its_transpose():
    op = Sz(0) + Sy(1) + Splus(0) * Sminus(1)            # non-Hermitian on purpose
    dense = _on(0, "z") + _on(1, "y") + _on(0, "+") @ _on(1, "-")
    pack = lambda s: np.asarray(_spin_index(s))
    mat, order = build_sparse_hamiltonian(SPIN_STATE, op, pack)
    np.testing.assert_array_equal(order, np.arange(4))
    np.testing.assert_allclose(mat.toarray(), dense, atol=1e-12)


# ------------------------------------------------------------- fermions -----
# Two sites and two bands: modes m = band * Ns + site, i.e. (0 up, 1 up,
# 0 down, 1 down), the whole 16-state Fock space. Index sum_m n_m 2^(3-m),
# so the dense matrices are Kronecker products with mode 0 as the left factor.

FERMION_BASIS = np.array(
    [[(k >> (3 - m)) & 1 for m in range(4)] for k in range(16)], dtype=np.int32)
FERMION_STATE = FermionState(occupations=jnp.asarray(FERMION_BASIS), Ne=2, lattice=LAT)
FERMION_AMPS = RNG.normal(size=16) + 1j * RNG.normal(size=16)


def _fermion_index(state):
    occ = jnp.atleast_2d(state.occupations)
    return occ @ (2 ** jnp.arange(3, -1, -1))


_A = np.array([[0, 1.0], [0, 0]])                      # annihilates |1>
_Z = np.diag([1.0, -1.0])                              # Jordan-Wigner string


def _c(m):
    """c_m on 4 modes: Z on every mode before m, as the tachys operators count."""
    factors = [_Z] * m + [_A] + [np.eye(2)] * (3 - m)
    out = factors[0]
    for f in factors[1:]:
        out = np.kron(out, f)
    return out


def _cd(m):
    return _c(m).T


UP0, UP1, DN0, DN1 = 0, 1, 2, 3
ALPHA = 0.3 + 0.7j

FERMION_CASES = {
    # real and symmetric: the same in either convention, checks the references
    "Nup(0) * Ndn(1)":                 (lambda: Nup(0) * Ndn(1),
                                        (_cd(UP0) @ _c(UP0)) @ (_cd(DN1) @ _c(DN1))),
    "HoppingUp(0, 1, real)":           (lambda: HoppingUp(0, 1, coupling=-1.0),
                                        -(_cd(UP0) @ _c(UP1) + _cd(UP1) @ _c(UP0))),
    # complex or non-symmetric: <psi|O|psi> only in the row convention
    "Cup_dag(0)":                      (lambda: Cup_dag(0), _cd(UP0)),
    "Cdn(1)":                          (lambda: Cdn(1), _c(DN1)),
    "Cup_dag(0) * Cup(1)":             (lambda: Cup_dag(0) * Cup(1), _cd(UP0) @ _c(UP1)),
    "Cdn_dag(1) * Cup(0)":             (lambda: Cdn_dag(1) * Cup(0), _cd(DN1) @ _c(UP0)),
    "Nup(1) * Cup_dag(1) * Cup(0)":    (lambda: Nup(1) * Cup_dag(1) * Cup(0),
                                        _cd(UP1) @ _c(UP1) @ _cd(UP1) @ _c(UP0)),
    "Cup_dag(1) * Ndn(0) * Cup(0)":    (lambda: Cup_dag(1) * Ndn(0) * Cup(0),
                                        _cd(UP1) @ (_cd(DN0) @ _c(DN0)) @ _c(UP0)),
    "HoppingUp(0, 1, complex)":        (lambda: HoppingUp(0, 1, coupling=ALPHA),
                                        ALPHA * _cd(UP0) @ _c(UP1)
                                        + np.conj(ALPHA) * _cd(UP1) @ _c(UP0)),
    "HoppingDown(1, 0, complex)":      (lambda: HoppingDown(1, 0, coupling=ALPHA),
                                        ALPHA * _cd(DN1) @ _c(DN0)
                                        + np.conj(ALPHA) * _cd(DN0) @ _c(DN1)),
}


@pytest.mark.parametrize("name", list(FERMION_CASES))
def test_fermion_expectation_is_psi_O_psi(name):
    make_op, dense = FERMION_CASES[name]
    psi = FERMION_AMPS / np.linalg.norm(FERMION_AMPS)
    expected = np.vdot(psi, dense @ psi)
    got = _exact_average(make_op(), FERMION_STATE, FERMION_AMPS, _fermion_index)
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_fermion_sparse_matrix_is_O_not_its_transpose():
    op = Nup(0) + HoppingUp(0, 1, coupling=ALPHA) + Cup_dag(0) * Cdn(1)
    dense = (_cd(UP0) @ _c(UP0)
             + ALPHA * _cd(UP0) @ _c(UP1) + np.conj(ALPHA) * _cd(UP1) @ _c(UP0)
             + _cd(UP0) @ _c(DN1))
    pack = lambda s: np.asarray(_fermion_index(s))
    mat, order = build_sparse_hamiltonian(FERMION_STATE, op, pack)
    np.testing.assert_array_equal(order, np.arange(16))
    np.testing.assert_allclose(mat.toarray(), dense, atol=1e-12)


# ---------------------------------------------- gradients and dynamics -----
# SR and TDVP take the local energies from the estimator, and their formulas,
# grad E = 2 Re <dO* dE_L> and Re(S) thetadot = Im <dO* dE_L>, hold for the
# row-based E_L = (H psi) / psi only. The models below are Hermitian but not
# real (a Dzyaloshinskii-Moriya term and a field along y; a complex hopping),
# so H != H^T and the old column convention would optimize and evolve under
# H^T = H* instead. The whole basis is the batch and the weights are |psi|^2,
# so the weighted optimizer path averages exactly. The wavefunction is a lookup
# table with real parameters, log psi(x) = a[x] + i b[x]: it spans the whole
# Hilbert space, so TDVP is exact and must reproduce i d/dt psi = H psi.

D, G = 0.7, -0.4
SPIN_H = (Sz(0) * Sz(1) + 0.5 * Splus(0) * Sminus(1) + 0.5 * Sminus(0) * Splus(1)
          + D * Sx(0) * Sy(1) + (-D) * Sy(0) * Sx(1) + G * Sy(0))
SPIN_H_DENSE = (_on(0, "z") @ _on(1, "z")
                + 0.5 * (_on(0, "+") @ _on(1, "-") + _on(0, "-") @ _on(1, "+"))
                + D * (_on(0, "x") @ _on(1, "y") - _on(0, "y") @ _on(1, "x"))
                + G * _on(0, "y"))

U = 2.5
FERMION_H = (HoppingUp(0, 1, coupling=ALPHA)
             + ALPHA * Cdn_dag(0) * Cdn(1) + np.conj(ALPHA) * Cdn_dag(1) * Cdn(0)
             + U * Nup(0) * Ndn(0) + U * Nup(1) * Ndn(1))
FERMION_H_DENSE = (ALPHA * (_cd(UP0) @ _c(UP1) + _cd(DN0) @ _c(DN1))
                   + np.conj(ALPHA) * (_cd(UP1) @ _c(UP0) + _cd(DN1) @ _c(DN0))
                   + U * (_cd(UP0) @ _c(UP0)) @ (_cd(DN0) @ _c(DN0))
                   + U * (_cd(UP1) @ _c(UP1)) @ (_cd(DN1) @ _c(DN1)))

MODELS = {
    "spins": (SPIN_H, SPIN_H_DENSE, SPIN_STATE, _spin_index, 4),
    "fermions": (FERMION_H, FERMION_H_DENSE, FERMION_STATE, _fermion_index, 16),
}


def _setup(name):
    """Lookup-table wavefunction, exact weights, and dense tangent vectors."""
    H, H_dense, state, index, n = MODELS[name]
    assert np.allclose(H_dense, H_dense.conj().T) and not np.allclose(H_dense, H_dense.T)
    rng = np.random.default_rng(1)
    params = {"a": jnp.asarray(0.6 * rng.normal(size=n)),
              "b": jnp.asarray(2.0 * rng.normal(size=n))}
    wf = WaveFunction(params=params,
                      apply_fn=lambda p, s: p["a"][index(s)] + 1j * p["b"][index(s)])
    log_psi = wf.apply_fn(wf.params, state)
    psi = np.asarray(jnp.exp(log_psi))
    p = np.abs(psi) ** 2 / np.sum(np.abs(psi) ** 2)
    E_L, _, _ = compute_expectation(H, wf, state, log_psi)

    flat, unravel = ravel_pytree(wf.params)
    log_psi_of = lambda f: wf.apply_fn(unravel(f), state)
    O = np.asarray(jax.jacfwd(lambda f: jnp.real(log_psi_of(f)))(flat)
                   + 1j * jax.jacfwd(lambda f: jnp.imag(log_psi_of(f)))(flat))
    return H, H_dense, state, wf, E_L, psi, p, O, flat, log_psi_of


@pytest.mark.parametrize("name", list(MODELS))
def test_sr_update_is_the_natural_gradient_of_psi_H_psi(name):
    H, H_dense, state, wf, E_L, psi, p, O, flat, log_psi_of = _setup(name)

    def energy(f):
        ps = jnp.exp(log_psi_of(f))
        return jnp.real(jnp.vdot(ps, H_dense @ ps) / jnp.vdot(ps, ps))

    dO = O - p @ O
    S = np.real(dO.conj().T @ (p[:, None] * dO))
    lam = 1e-2
    expected = np.linalg.solve(S + lam * np.eye(len(flat)), np.asarray(jax.grad(energy)(flat)))

    sr = SR(diag_shift=lam, mode="complex")
    updates, _ = sr(E_L, sr.init(wf.params), state, wf, weights=jnp.asarray(p * len(p)))
    np.testing.assert_allclose(ravel_pytree(updates)[0], expected, atol=1e-10)


@pytest.mark.parametrize("name", list(MODELS))
def test_tdvp_velocity_solves_the_schrodinger_equation(name):
    H, H_dense, state, wf, E_L, psi, p, O, flat, log_psi_of = _setup(name)

    tdvp = TDVP(mode="complex", rcond=1e-10)
    velocity, _ = tdvp(E_L, tdvp.init(wf.params), state, wf, weights=jnp.asarray(p * len(p)))
    dlog_psi_dt = O @ np.asarray(ravel_pytree(velocity)[0])

    # i d/dt psi = H psi, up to the norm and global phase TDVP leaves free:
    # d/dt log psi(x) + i (H psi)(x) / psi(x) is the same constant for every x.
    residual = dlog_psi_dt + 1j * (H_dense @ psi) / psi
    np.testing.assert_allclose(residual - p @ residual, 0, atol=1e-10)
