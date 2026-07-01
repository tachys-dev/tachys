import numpy as np
import jax.numpy as jnp

from ..spin_operators import Sminus, Splus, Sz, XYExchange

def heisenberg_lattice(lat, nn, J=1.0):
    """Heisenberg Hamiltonian on a generic Lattice.

    H = J * sum_{<i,j>} [ Sz_i Sz_j + (1/2)(S+_i S-_j + S-_i S+_j) ]

    Args:
        lat: Lattice object.
        nn:  Bond specifications. Each element is one of:
               (d1, d2)             — displacement in (a1,a2) units, b_from=0
               ((d1,d2), b_from)    — explicit source sublattice
               ((d1,d2), b_from, b_to) — explicit source and target sublattice
        J:   exchange coupling (positive = antiferromagnetic).
    """
    bond_src, bond_dst = [], []
    for spec in nn:
        if hasattr(spec[0], '__len__'):
            d      = spec[0]
            b_from = spec[1] if len(spec) > 1 else 0
            b_to   = spec[2] if len(spec) > 2 else None
        else:
            d, b_from, b_to = spec, 0, None
        s, t = lat.bonds(d, b_from=b_from, b_to=b_to)
        bond_src.append(s)
        bond_dst.append(t)
    src = np.concatenate(bond_src)
    dst = np.concatenate(bond_dst)

    H = J * Sz(src) * Sz(dst) 
    H = H + J/2 * XYExchange(i=src, j=dst)
    return H


def heisenberg_square_pbc_exchange(L, J=1.0):
    """Heisenberg Hamiltonian on an L×L square lattice with periodic boundary conditions.

    H = J * sum_{<i,j>} [ Sz_i Sz_j + (1/2)(S+_i S-_j + S-_i S+_j) ]

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

    H = XYExchange(i=is_, j=js_, coupling= J / 2)
    H = H + J * Sz(is_) * Sz(js_)

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
