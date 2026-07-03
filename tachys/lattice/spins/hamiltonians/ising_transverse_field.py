import numpy as np

from ..spin_operators import Sx, Sz


def ising_transverse_field_hamiltonian(lat, nn, h=1.0):
    """Transverse-field Ising Hamiltonian on a generic Lattice.

    H = -sum_{<i,j>} J_ij * sigma_z_i sigma_z_j - h * sum_i sigma_x_i

    J and h are in units of the Pauli matrix convention (sigma = 2S).

    Args:
        lat: Lattice object.
        nn:  Bond specifications, each pairing a displacement with its coupling:
               ((d1, d2), J_ij)                     — b_from=0
               ((d1, d2), J_ij, b_from)              — explicit source sublattice
               ((d1, d2), J_ij, b_from, b_to)        — explicit source and target sublattice
             List further shells (e.g. next-nearest-neighbour bonds) as additional
             specs with their own displacement/coupling/sublattices.
        h: Transverse field strength.
    """
    bond_src, bond_dst, bond_J = [], [], []
    for d, J_ij, *rest in nn:
        b_from = rest[0] if len(rest) > 0 else 0
        b_to   = rest[1] if len(rest) > 1 else None
        s, t = lat.bonds(d, b_from=b_from, b_to=b_to)
        bond_src.append(s)
        bond_dst.append(t)
        bond_J.append(np.full(len(s), J_ij, dtype=float))
    src = np.concatenate(bond_src)
    dst = np.concatenate(bond_dst)
    coupling = np.concatenate(bond_J)

    # sigma_z = 2*Sz (Sz has eigenvalues ±1/2), sigma_x = 2*Sx (Sx has off-diagonal 1/2)
    H = -4 * Sz(src, coupling=coupling) * Sz(dst, coupling=np.ones_like(coupling))
    H = H + (-2 * h) * Sx(tuple(range(lat.Ns)))

    return H


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
    H = -4 * J * Sz(is_) * Sz(js_)
    H = H + (-2 * h) * Sx(sites)

    return H
