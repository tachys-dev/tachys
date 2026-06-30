import numpy as np

from ..spin_operators import Sx, Sz


def ising_transverse_field_square_pbc(L, J=1.0, h=1.0):
    """Transverse-field Ising model on an L×L square lattice with periodic boundary conditions.

    H = -J * sum_{<i,j>} Sz_i Sz_j - h * sum_i Sx_i

    Sites are indexed row-major: site(x, y) = x*L + y, with x in [0,L) and y in [0,L).
    """
    bonds = [
        (x * L + y, x * L + (y + 1) % L)
        for x in range(L) for y in range(L)
    ] + [
        (x * L + y, ((x + 1) % L) * L + y)
        for x in range(L) for y in range(L)
    ]
    is_, js_ = zip(*bonds)
    sites = tuple(range(L * L))
    
    H = -J * Sz(np.array(is_)) * Sz(np.array(js_))
    H = H + (-h) * Sx(np.array(sites))

    return H


def ising_transverse_field_chain_pbc(L, J=1.0, h=1.0):
    """Transverse-field Ising model on a 1D chain with periodic boundary conditions.

    H = -J * sum_i Sz_i Sz_{i+1} - h * sum_i Sx_i
    """
    is_ = tuple(range(L))
    js_ = tuple((i + 1) % L for i in range(L))
    sites = is_

    H = -J * Sz(is_) * Sz(js_)
    H = H + (-h) * Sx(sites)

    return H
