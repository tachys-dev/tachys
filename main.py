import jax.numpy as jnp
from nuxem.lattice.exact_diag import exact_diag, spins_hilbert_space
from nuxem.lattice.spins.hamiltonians.heisenberg import heisenberg_square_pbc, heisenberg_square_pbc_exchange
from nuxem.lattice.operator.base import DiagonalResult, OffdiagonalResult, DiagOffdiagResult
from nuxem.lattice.spins.spin_state import SpinState
import scipy
from scipy.sparse.linalg import eigsh
import numpy as np

L = 4  # 2×2 square lattice (4 sites)
N = L*L

H = heisenberg_square_pbc_exchange(L, J=1.0)
all_states = spins_hilbert_space(N)
state_full_hilbert = SpinState(spins=jnp.array(all_states, dtype=jnp.int8), Ns=N)

def pack(state):
    bits = (np.asarray(state.spins) + 1) // 2
    return (bits * 2 ** np.arange(state.spins.shape[-1])).sum(axis=-1)

eigenvalues, _ = exact_diag(state_full_hilbert, H, pack, k=1)

print(eigenvalues[0] / N)