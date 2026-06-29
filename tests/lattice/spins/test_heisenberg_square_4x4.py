import jax.numpy as jnp
import numpy as np
import pytest

from nuxem.lattice.exact_diag import spins_hilbert_space, exact_diag
from nuxem.lattice.spins.hamiltonians.heisenberg import heisenberg_square_pbc
from nuxem.lattice.spins.spin_state import SpinState

L = 4
N = L * L
# Reference: exact diagonalisation of 4×4 Heisenberg model (J=1, PBC)
# E0/N = -0.70178020...
E0_PER_SITE = -0.7017802005268037


@pytest.fixture(scope="module")
def ground_state_energy():
    H = heisenberg_square_pbc(L, J=1.0)
    all_states = spins_hilbert_space(N)
    state_full_hilbert = SpinState(spins=jnp.array(all_states, dtype=jnp.int8), Ns=N)

    def pack(state):
        bits = (np.asarray(state.spins) + 1) // 2
        return (bits * 2 ** np.arange(state.spins.shape[-1])).sum(axis=-1)
    
    hilbert_dim = 2**N
    
    eigenvalues, _ = exact_diag(state_full_hilbert, H, hilbert_dim, pack, k=1)
    return float(eigenvalues[0])


def test_ground_state_energy_per_site(ground_state_energy):
    assert ground_state_energy / N == pytest.approx(E0_PER_SITE, rel=1e-12)
