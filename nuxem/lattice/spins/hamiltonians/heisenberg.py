from ..spin_operators import Sminus, Splus, Sz


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
