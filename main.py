import jax.numpy as jnp
from tachys.lattice.exact_diag import exact_diag, fermions_hilbert_space, spins_hilbert_space
from tachys.lattice.fermions.fermion_state import FermionState
from tachys.lattice.fermions.hamiltonians.hubbard import hubbard_square_pbc
from tachys.lattice.spins.hamiltonians.heisenberg import heisenberg_square_pbc, heisenberg_square_pbc_exchange
from tachys.lattice.operator.base import DiagonalResult, OffdiagonalResult, DiagOffdiagResult
from tachys.lattice.spins.spin_state import SpinState
import scipy
from scipy.sparse.linalg import eigsh
import numpy as np

L = 4
N = L*L
Ne = 4

H = hubbard_square_pbc(L, U=8)
all_states = fermions_hilbert_space(N, Ne, 2)
state_full_hilbert = FermionState(occupations=jnp.array(all_states, dtype=jnp.int8), Ns=N, Ne=Ne)

def pack(state):
    occupations = state.occupations
    return (occupations * 2 ** np.arange(occupations.shape[-1])).sum(axis=-1)

eigenvalues, _ = exact_diag(state_full_hilbert, H, pack, k=1)

print(eigenvalues[0])