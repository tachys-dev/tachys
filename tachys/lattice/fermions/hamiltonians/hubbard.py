import numpy as np

from ..fermion_operators import Cup, Cup_dag, Cdn, Cdn_dag, HoppingUp, HoppingDown, Nup, Ndn


def hubbard_hamiltonian(lat, nn, U):
    """Hubbard Hamiltonian on a generic Lattice.

    H = sum_{<i,j>,σ} -t_ij (c†_{i,σ} c_{j,σ} + h.c.) + U * sum_i n_{i,↑} n_{i,↓}

    Args:
        lat: Lattice object.
        nn:  Bond specifications, each pairing a displacement with its hopping
             amplitude:
               ((d1, d2), t_ij)                     — b_from=0
               ((d1, d2), t_ij, b_from)              — explicit source sublattice
               ((d1, d2), t_ij, b_from, b_to)        — explicit source and target sublattice
             List further shells (e.g. next-nearest-neighbour hopping) as additional
             specs with their own displacement/amplitude/sublattices.
        U:   on-site interaction strength.
    """
    bond_src, bond_dst, bond_t = [], [], []
    for d, t_ij, *rest in nn:
        b_from = rest[0] if len(rest) > 0 else 0
        b_to   = rest[1] if len(rest) > 1 else None
        s, t = lat.bonds(d, b_from=b_from, b_to=b_to)
        bond_src.append(s)
        bond_dst.append(t)
        bond_t.append(np.full(len(s), t_ij, dtype=float))
    src = np.concatenate(bond_src)
    dst = np.concatenate(bond_dst)
    hop = np.concatenate(bond_t)

    H = HoppingUp(src, dst, coupling=-hop) + HoppingDown(src, dst, coupling=-hop)

    sites = np.arange(lat.Ns)
    H = H + Nup(sites, coupling=np.full(lat.Ns, U)) * Ndn(sites, coupling=np.ones(lat.Ns))
    return H


def hubbard_square_pbc(L, t=1.0, U=0.0):
    """Hubbard model on an L×L square lattice with periodic boundary conditions.

    H = -t * sum_{<i,j>,σ} (c†_{i,σ} c_{j,σ} + h.c.) + U * sum_i n_{i,↑} n_{i,↓}

    Sites are indexed row-major: site(x, y) = x*L + y, with x in [0,L) and y in [0,L).
    """
    H = None

    for x in range(L):
        for y in range(L):
            i = x * L + y

            interaction = U * Nup(i) * Ndn(i)
            H = interaction if H is None else H + interaction

            for j in (x * L + (y + 1) % L, ((x + 1) % L) * L + y):
                hopping = (  - t * Cup_dag(i) * Cup(j)
                           + - t * Cup_dag(j) * Cup(i)
                           + - t * Cdn_dag(i) * Cdn(j)
                           + - t * Cdn_dag(j) * Cdn(i))
                H = H + hopping

    return H
