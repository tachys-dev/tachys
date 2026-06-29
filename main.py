import jax.numpy as jnp
from nuxem.lattice.exact_diag import spins_hilbert_space
from nuxem.lattice.hamiltonians.heisenberg import heisenberg_square_pbc
from nuxem.lattice.operator.base import DiagonalResult, OffdiagonalResult, DiagOffdiagResult
from nuxem.lattice.state.spins import SpinState
import scipy
from scipy.sparse.linalg import eigsh
import numpy as np

L = 4  # 2×2 square lattice (4 sites)
H = heisenberg_square_pbc(L, J=1.0)

all_states = spins_hilbert_space(L*L)
all_states = all_states[all_states.sum(-1)==0]
state_full_hilbert = SpinState(spins=jnp.array(all_states, dtype=jnp.int8), Ns=L*L)
hilbert_dim = state_full_hilbert.spins.shape[0]

result = H(state_full_hilbert)

assert isinstance(result, DiagOffdiagResult)

diag_me = result.diagonal.matrix_element          # (n_diag_terms, n_states)
offdiag_states = result.offdiagonal.connected_states
offdiag_mask = result.offdiagonal.mask            # (n_offdiag_terms, n_states)
offdiag_me = result.offdiagonal.matrix_element    # (n_offdiag_terms, n_states)

packbits = lambda x_bits, Ns : (x_bits*2**np.arange(Ns)).sum(axis=-1)
packspins = lambda x_bits, Ns : packbits((x_bits + 1)//2, Ns)

# off diagonal terms
cols = packspins(offdiag_states.spins, L*L)
rows = np.broadcast_to(np.arange(hilbert_dim)[None], cols.shape)
cols, rows = cols.flatten(), rows.flatten()
values = (offdiag_mask * offdiag_me).flatten()

# diagonal terms
cols = np.concatenate((np.arange(hilbert_dim), cols))
rows = np.concatenate((np.arange(hilbert_dim), rows))
values = np.concatenate((diag_me.sum(0), values))

H = scipy.sparse.coo_array((values, (rows, cols)), shape=(hilbert_dim, hilbert_dim))
H.eliminate_zeros()
H = 0.5 * (H + H.T)
H = H.tocsr()
eigval, v = eigsh(H, k=1, which="SA", return_eigenvectors=True)

print('Eigenvalues : ')

for i in range(len(eigval)):
    print(i,')  ', eigval[i]/L**2)