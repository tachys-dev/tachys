import numpy as np

from ..spin_operators import Sx, Sz


def ising_transverse_field_square_pbc(L, J=1.0, h=1.0):
    """Transverse-field Ising model on an L×L square lattice with periodic boundary conditions.

    H = -J * sum_{<i,j>} sigma_z_i sigma_z_j - h * sum_i sigma_x_i

    J and h are in units of the Pauli matrix convention (sigma = 2S).
    The 1D chain critical point is at J = h.

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

    # sigma_z = 2*Sz (Sz has eigenvalues ±1/2), sigma_x = 2*Sx (Sx has off-diagonal 1/2)
    H = -4 * J * Sz(np.array(is_)) * Sz(np.array(js_))
    H = H + (-2 * h) * Sx(np.array(sites))

    return H


def ising_transverse_field_chain_pbc(L, J=1.0, h=1.0):
    """Transverse-field Ising model on a 1D chain with periodic boundary conditions.

    H = -J * sum_i sigma_z_i sigma_z_{i+1} - h * sum_i sigma_x_i

    J and h are in units of the Pauli matrix convention (sigma = 2S).
    The critical point is at J = h.
    """
    is_ = tuple(range(L))
    js_ = tuple((i + 1) % L for i in range(L))
    sites = is_

    # sigma_z = 2*Sz (Sz has eigenvalues ±1/2), sigma_x = 2*Sx (Sx has off-diagonal 1/2)
    H = -4 * J * Sz(is_) * Sz(js_)
    H = H + (-2 * h) * Sx(sites)

    return H
