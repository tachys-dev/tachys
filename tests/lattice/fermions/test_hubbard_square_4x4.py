import jax.numpy as jnp
import numpy as np
import pytest

from tachys.lattice.exact_diag import fermions_hilbert_space, exact_diag
from tachys.lattice.fermions.fermion_state import FermionState
from tachys.lattice.fermions.hamiltonians.hubbard import hubbard_square_pbc

L = 4
Ns = L * L
Ne = 4
# Reference: E. Dagotto et al., Phys. Rev. B 45, 10741 (1992)
# Ground state energies for the 4×4 Hubbard model with Ne=4 electrons (t=1)
E0_BY_U = {4.0: -11.53029, 8.0: -11.32150}


def _pack(state):
    occ = np.asarray(state.occupations)
    return (occ * 2 ** np.arange(occ.shape[-1])).sum(axis=-1)


@pytest.fixture(scope="module")
def hubbard_ground_state_energies():
    all_states = fermions_hilbert_space(Ns, Ne)
    state_full = FermionState(
        occupations=jnp.array(all_states, dtype=jnp.int8), Ns=Ns, Ne=Ne
    )
    results = {}
    for U in (4.0, 8.0):
        H = hubbard_square_pbc(L, t=1.0, U=U)
        eigenvalues, _ = exact_diag(state_full, H, _pack, k=1)
        results[U] = float(eigenvalues[0])
    return results


def test_ground_state_energy_U4(hubbard_ground_state_energies):
    assert hubbard_ground_state_energies[4.0] == pytest.approx(E0_BY_U[4.0], abs=5e-5)


def test_ground_state_energy_U8(hubbard_ground_state_energies):
    assert hubbard_ground_state_energies[8.0] == pytest.approx(E0_BY_U[8.0], abs=5e-5)
