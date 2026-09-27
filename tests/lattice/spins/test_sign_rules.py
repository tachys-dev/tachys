import jax.numpy as jnp
import numpy as np
import pytest

from tachys.lattice.exact_diag import spins_hilbert_space, exact_diag
from tachys.lattice.spins.hamiltonians.heisenberg import heisenberg_hamiltonian
from tachys.lattice.spins.sign_rules import MSR_log_phase_square
from tachys.lattice.spins.spin_state import SpinState
from tachys.lattice.lattice_database import square


def _pack(state):
    bits = (np.asarray(state.spins) + 1) // 2
    return (bits * 2 ** np.arange(state.spins.shape[-1])).sum(axis=-1)


@pytest.mark.parametrize("shape", [(4, 4), (4, 2), (2, 4), (6, 4), (3, 5)])
def test_msr_square_sublattice_matches_lattice(shape):
    """Site s is on sublattice A iff row + column of s is even, with rows and
    columns read from the Lattice: flipping only spin s gives the phase i*pi
    on A and 0 on B."""
    lat = square(shape=shape)
    i, j = lat.site_coords[:, 0], lat.site_coords[:, 1]
    on_A = (i + j) % 2 == 0

    spins = np.ones((lat.Ns, lat.Ns), dtype=np.int8)
    np.fill_diagonal(spins, -1)                      # configuration s: only spin s down
    phase = MSR_log_phase_square(jnp.asarray(spins), lat.L)

    np.testing.assert_allclose(np.asarray(phase), np.where(on_A, 1j * np.pi, 0.0))


# (Lx, Ly, pbc_x, pbc_y): bipartite clusters, Lx != Ly except the first
BIPARTITE_CLUSTERS = [
    (4, 4, True, True),
    (4, 2, True, True),
    (2, 4, True, True),
    (4, 3, True, False),
    (3, 4, False, True),
]


@pytest.mark.parametrize("Lx, Ly, pbc_x, pbc_y", BIPARTITE_CLUSTERS)
def test_msr_square_heisenberg_ground_state(Lx, Ly, pbc_x, pbc_y):
    """Marshall: the ground state of the nearest-neighbour Heisenberg model is
    psi(s) = (-1)^{N_down^A} |psi(s)|, so its sign times exp(MSR) is constant."""
    lat = square(shape=(Lx, Ly), pbc_x=pbc_x, pbc_y=pbc_y)
    H = heisenberg_hamiltonian(lat, nn=[((1, 0), 1.0), ((0, 1), 1.0)])
    all_states = spins_hilbert_space(lat.Ns)
    state = SpinState(spins=jnp.array(all_states, dtype=jnp.int8), lattice=lat)
    _, eigenvectors = exact_diag(state, H, _pack, k=1)

    psi = eigenvectors[:, 0]                         # row r of all_states has _pack = r
    msr_sign = np.real(np.exp(np.asarray(MSR_log_phase_square(state.spins, lat.L))))
    support = np.abs(psi) > 1e-8 * np.abs(psi).max()
    signs = np.sign(psi[support]) * msr_sign[support]

    assert np.all(signs == signs[0])
