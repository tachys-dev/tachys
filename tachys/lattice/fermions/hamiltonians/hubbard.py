from ..fermion_operators import Cup, Cup_dag, Cdn, Cdn_dag, Nup, Ndn

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
