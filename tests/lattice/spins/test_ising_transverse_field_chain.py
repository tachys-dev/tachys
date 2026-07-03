import jax.numpy as jnp
import numpy as np
import pytest

from tachys.lattice.exact_diag import spins_hilbert_space, exact_diag
from tachys.lattice.spins.hamiltonians.ising_transverse_field import ising_transverse_field_hamiltonian
from tachys.lattice.spins.spin_state import SpinState
from tachys.lattice.lattice_database import chain

# ---- reference values (Pauli convention sigma=2S) ------------------------
# 10-site chain at the critical point (J=h=1), PBC: E0/N = -1.2784906442999309
CHAIN_10_H1_PBC_E0_PER_SITE = -1.2784906442999309

# 10-site chain at the critical point (J=h=1), OBC: E0/N = -1.2381489999654734
CHAIN_10_H1_OBC_E0_PER_SITE = -1.2381489999654734


def _pack(state):
    bits = (np.asarray(state.spins) + 1) // 2
    return (bits * 2 ** np.arange(state.spins.shape[-1])).sum(axis=-1)


def _run_ed(H, lat):
    all_states = spins_hilbert_space(lat.Ns)
    state = SpinState(spins=jnp.array(all_states, dtype=jnp.int8), lattice=lat)
    eigenvalues, _ = exact_diag(state, H, _pack, k=1)
    return float(eigenvalues[0])


@pytest.fixture(scope="module")
def chain_10_h1_pbc_energy():
    L = 10
    lat = chain(L, pbc=True)
    H = ising_transverse_field_hamiltonian(lat, nn=[((1, 0), 1.0)], h=1.0)
    return _run_ed(H, lat)


@pytest.fixture(scope="module")
def chain_10_h1_obc_energy():
    L = 10
    lat = chain(L, pbc=False)
    H = ising_transverse_field_hamiltonian(lat, nn=[((1, 0), 1.0)], h=1.0)
    return _run_ed(H, lat)


def test_chain_10_h1_pbc_ground_state_energy(chain_10_h1_pbc_energy):
    N = 10
    assert chain_10_h1_pbc_energy / N == pytest.approx(CHAIN_10_H1_PBC_E0_PER_SITE, rel=1e-12)


def test_chain_10_h1_obc_ground_state_energy(chain_10_h1_obc_energy):
    N = 10
    assert chain_10_h1_obc_energy / N == pytest.approx(CHAIN_10_H1_OBC_E0_PER_SITE, rel=1e-12)
