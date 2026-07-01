import numpy as np
import jax.numpy as jnp

from ..spin_operators import Sminus, Splus, Sz, XYExchange

def heisenberg_hamiltonian(lat, nn):
    """Heisenberg Hamiltonian on a generic Lattice.

    H = sum_{<i,j>} J_ij * [ Sz_i Sz_j + (1/2)(S+_i S-_j + S-_i S+_j) ]

    Args:
        lat: Lattice object.
        nn:  Bond specifications, each pairing a displacement with its coupling:
               ((d1, d2), J_ij)                     — b_from=0
               ((d1, d2), J_ij, b_from)              — explicit source sublattice
               ((d1, d2), J_ij, b_from, b_to)        — explicit source and target sublattice
             List further shells (e.g. next-nearest-neighbour bonds) as additional
             specs with their own displacement/coupling/sublattices.
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

    H = Sz(src, coupling=coupling) * Sz(dst, coupling=np.ones_like(coupling))
    H = H + XYExchange(i=src, j=dst, coupling=coupling / 2)
    return H

def heisenberg_square_pbc(L, J=1.0):
    """Heisenberg Hamiltonian on an L×L square lattice with periodic boundary conditions.

    H = J * sum_{<i,j>} [ Sz_i Sz_j + (1/2)(S+_i S-_j + S-_i S+_j) ]

    Sites are indexed row-major: site(x, y) = x*L + y, with x in [0,L) and y in [0,L).
    """
    H = None

    for x in range(L):
        for y in range(L):
            i = x * L + y
            for j in (x * L + (y + 1) % L, ((x + 1) % L) * L + y):
                bond = J*Sz(i)* Sz(j) + J/2 * Splus(i) * Sminus(j) + J/2 * Sminus(i) * Splus(j)
                H = bond if H is None else H + bond

    return H
