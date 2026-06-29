import jax.numpy as jnp
from nuxem.lattice.exact_diag import exact_diag, fermions_hilbert_space, spins_hilbert_space
from nuxem.lattice.fermions.fermion_state import FermionState
from nuxem.lattice.fermions.hamiltonians.hubbard import hubbard_square_pbc
from nuxem.lattice.spins.hamiltonians.heisenberg import heisenberg_square_pbc, heisenberg_square_pbc_exchange
from nuxem.lattice.operator.base import DiagonalResult, OffdiagonalResult, DiagOffdiagResult
from nuxem.lattice.spins.spin_state import SpinState
import scipy
from scipy.sparse.linalg import eigsh
import numpy as np

L = 4  # 2×2 square lattice (4 sites)
N = L*L
Ne = 4
hilbert_dim = 4**N

H = hubbard_square_pbc(L, U=8)
all_states = fermions_hilbert_space(N, Ne, 2)
state_full_hilbert = FermionState(occupations=jnp.array(all_states, dtype=jnp.int8), Ns=N, Ne=Ne)

def pack(state):
    occupations = state.occupations
    return (occupations * 2 ** np.arange(occupations.shape[-1])).sum(axis=-1)

eigenvalues, _ = exact_diag(state_full_hilbert, H, hilbert_dim, pack, k=1)

print(eigenvalues[0])