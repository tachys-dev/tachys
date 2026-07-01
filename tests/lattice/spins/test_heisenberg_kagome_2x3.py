import jax.numpy as jnp
import numpy as np
import pytest

from tachys.lattice.exact_diag import spins_hilbert_space, exact_diag
from tachys.lattice.spins.hamiltonians.heisenberg import heisenberg_hamiltonian
from tachys.lattice.spins.spin_state import SpinState
from tachys.lattice.lattice_database import kagome

# 2x3 Kagome (Lx=2, Ly=3): N = 3*2*3 = 18 sites, J=1, PBC
# E0/N = -0.4471261540856232
KAGOME_2x3_E0_PER_SITE = -0.4471261540856232

# Six undirected NN bond types for the Kagome lattice.
# Each spec is ((d1, d2), J_ij, b_from, b_to): a bond connects sublattice
# b_from in cell C to sublattice b_to in cell C + (d1, d2) (whole-cell
# displacement), with coupling J_ij.
KAGOME_NN = [
    ((0, 0),  1.0, 0, 1),   # b0 → b1, same cell
    ((0, 0),  1.0, 0, 2),   # b0 → b2, same cell
    ((0, 0),  1.0, 1, 2),   # b1 → b2, same cell
    ((1, 0),  1.0, 1, 0),   # b1 → b0, +a1 cell
    ((0, 1),  1.0, 2, 0),   # b2 → b0, +a2 cell
    ((1, -1), 1.0, 1, 2),   # b1 → b2, +a1−a2 cell
]


def _pack(state):
    bits = (np.asarray(state.spins) + 1) // 2
    return (bits * 2 ** np.arange(state.spins.shape[-1])).sum(axis=-1)


@pytest.fixture(scope="module")
def kagome_2x3_energy():
    lat = kagome(shape=(2, 3))
    H = heisenberg_hamiltonian(lat, nn=KAGOME_NN)
    all_states = spins_hilbert_space(lat.Ns)
    state = SpinState(spins=jnp.array(all_states, dtype=jnp.int8), Ns=lat.Ns)
    eigenvalues, _ = exact_diag(state, H, _pack, k=1)
    return float(eigenvalues[0])


def test_kagome_2x3_n_bonds(kagome_2x3_energy):
    lat = kagome(shape=(2, 3))
    src_all, dst_all = [], []
    for d, J_ij, b_from, b_to in KAGOME_NN:
        s, t = lat.bonds(d, b_from=b_from, b_to=b_to)
        src_all.append(s); dst_all.append(t)
    n_bonds = sum(len(s) for s in src_all)
    assert n_bonds == 6 * 2 * 3  # 6 bond types × Lx × Ly


def test_kagome_2x3_ground_state_energy(kagome_2x3_energy):
    N = 3 * 2 * 3
    assert kagome_2x3_energy / N == pytest.approx(KAGOME_2x3_E0_PER_SITE, rel=1e-10)
