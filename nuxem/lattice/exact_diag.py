import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from itertools import combinations
from math import comb

from jax import config
config.update("jax_enable_x64", True)


def exact_diag(state_full_hilbert, H, hilbert_dim, pack, k=1):
    """
    Build the sparse Hamiltonian matrix and return the lowest k eigenvalues
    and eigenvectors.

    Parameters
    ----------
    state_full_hilbert : SpinState
        SpinState whose .spins rows enumerate every basis state.
    H : operator
        Callable that accepts a SpinState and returns a DiagOffdiagResult.
    pack : callable
        Maps a SpinState to a 1-D integer array of basis indices.
    k : int
        Number of lowest eigenvalues to compute (default 1).

    Returns
    -------
    eigenvalues : np.ndarray, shape (k,)
    eigenvectors : np.ndarray, shape (hilbert_dim, k)
    """
    diag_idx = pack(state_full_hilbert)

    result = H(state_full_hilbert)

    diag_me = result.diagonal.matrix_element          # (n_diag_terms, dim)
    offdiag_states = result.offdiagonal.connected_states
    offdiag_mask = result.offdiagonal.mask             # (n_offdiag_terms, dim)
    offdiag_me = result.offdiagonal.matrix_element     # (n_offdiag_terms, dim)

    # off-diagonal entries
    off_cols = np.asarray(pack(offdiag_states))        # (n_offdiag_terms, dim)
    off_rows = np.broadcast_to(diag_idx[None], off_cols.shape)
    off_vals = np.asarray((offdiag_mask * offdiag_me).flatten())
    off_cols, off_rows = off_cols.flatten(), off_rows.flatten()

    # diagonal entries
    diag_vals = np.asarray(diag_me.sum(0))

    rows = np.concatenate((diag_idx, off_rows))
    cols = np.concatenate((diag_idx, off_cols))
    vals = np.concatenate((diag_vals, off_vals))

    mat = scipy.sparse.coo_array((vals, (rows, cols)), shape=(hilbert_dim, hilbert_dim))
    mat.eliminate_zeros()
    mat = 0.5 * (mat + mat.T)
    mat = mat.tocsr()

    eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(mat, k=k, which="SA")
    return eigenvalues, eigenvectors


def spins_hilbert_space(N: int, values=(-1, 1)):
    """
    Generate the full Hilbert space basis for N spin-1/2 sites in the σ^z basis.

    Parameters
    ----------
    N : int
        Number of spins.
    values : tuple, optional
        Pair (down_value, up_value). Defaults to (-1, +1).
        Use (0, 1) if you prefer bits.

    Returns
    -------
    np.ndarray
        Array with shape (2**N, N). Each row is a basis state.
    """
    if N < 0:
        raise ValueError("N must be non-negative.")
    if N == 0:
        return np.empty((1, 0), dtype=int)

    n = 1 << N  # 2**N
    down, up = values

    ints = np.arange(n, dtype=np.uint64)[:, None]  # shape (2**N, 1)
    shifts = np.arange(N, dtype=np.uint64)         # shape (N,)
    bits = (ints >> shifts) & 1                    # shape (2**N, N)

    states = np.where(bits == 1, up, down)
    return states

def fermions_hilbert_space(Ns: int, Ne: int, Nbands: int = 2):
    """
    Generate the full Hilbert space basis for Ne electrons on Ns sites.

    Parameters
    ----------
    Ns : int
        Number of lattice sites.
    Ne : int
        Number of electrons.
    Nbands : int, optional
        Number of bands (default 2 for spin-up/spin-down).

    Returns
    -------
    np.ndarray
        Array with shape (C(Nbands*Ns, Ne), Nbands*Ns). Each row is a basis
        state; entry [r, b*Ns+i] is 1 if band b, site i is occupied.
    """
    n_modes = Nbands * Ns
    dim = comb(n_modes, Ne)
    states = np.zeros((dim, n_modes), dtype=np.int32)
    for idx, occupied in enumerate(combinations(range(n_modes), Ne)):
        states[idx, list(occupied)] = 1
    return states