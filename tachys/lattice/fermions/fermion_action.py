from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from flax import struct

from tachys.montecarlo import _BaseAction
from tachys.lattice.spins.spin_action import exchange_spins


class FermionSpinExchange(_BaseAction):
    """Proposes exchanging the local spin between two singly-occupied sites
    within `max_dist` lattice-neighbour shells of each other: an up electron
    at site i becomes a down electron at the same site i, and a down
    electron at a neighbouring site j becomes an up electron at that same
    site j (i and j are drawn, in random order, from a candidate bond
    precomputed from the lattice geometry). No electron actually hops — each
    flips its own band in place — but the net effect on sites i and j is the
    same as swapping their (opposite) spin orientations.

    Combined, the move conserves Nup and Ndn (one electron moves up-band ->
    down-band at site i, one moves down-band -> up-band at site j). Requires
    site i to currently hold an up electron with its down slot empty, and
    site j to hold a down electron with its up slot empty; `allowed_move` is
    False otherwise (occupied-source / empty-destination guard).

    The bond is drawn uniformly from the fixed, precomputed `bonds` list
    (not by picking a site and then one of its neighbours), so the proposal
    stays symmetric and log_prob_correction = 0 even with OBC, where boundary
    sites have fewer neighbours than bulk sites (a site-then-neighbour scheme
    would implicitly weight by 1/degree(site) and break detailed balance
    there). Unlike BondExchange, the random choice is not restricted to only
    the currently-valid bonds — this move's validity condition (occupied +
    empty on both sides of the bond) is stronger than BondExchange's, so a
    walker could plausibly have zero valid bonds at some step, which would
    make a validity-weighted jax.random.choice ill-defined; proposing
    uniformly and rejecting invalid draws via `allowed_move` sidesteps that.

    Attributes:
        max_dist: Maximum bond distance (in lattice shells) between the two sites.
        bonds: [N_bonds, 2] candidate (site_i, site_j) pairs, precomputed from
            the lattice geometry up to `max_dist` shells.
    """

    max_dist: int = struct.field(pytree_node=False)
    bonds: tuple = struct.field(pytree_node=False)

    @classmethod
    def create(cls, lattice, max_dist=1):
        bonds = [np.column_stack((src, dst)) for _, src, dst in lattice.shells(max_dist)]
        bonds = np.vstack(bonds)
        return cls(max_dist=max_dist, bonds=tuple(map(tuple, bonds.tolist())))

    def __call__(self, key, state):
        Ns = state.Ns
        N_mc_local = state.occupations.shape[0]
        chain_idx = jnp.arange(N_mc_local)

        bonds_arr = jnp.asarray(self.bonds)
        N_bonds = bonds_arr.shape[0]

        rands = jax.vmap(partial(jax.random.uniform, shape=(2,)))(key)
        bond_index = (rands[:, 0] * N_bonds).astype(int)
        swap = rands[:, 1] < 0.5

        site_i = bonds_arr[bond_index, 0]
        site_j = bonds_arr[bond_index, 1]

        index_up = jnp.where(swap, site_j, site_i)
        index_dn = jnp.where(swap, site_i, site_j) + Ns

        occupied = (state.occupations[chain_idx, index_up] != 0) & (state.occupations[chain_idx, index_dn] != 0)
        empty_dest = (state.occupations[chain_idx, index_up + Ns] == 0) & (state.occupations[chain_idx, index_dn - Ns] == 0)
        allowed_move = occupied & empty_dest

        new_occupations = jax.vmap(exchange_spins)(state.occupations, index_up, index_up + Ns)
        new_occupations = jax.vmap(exchange_spins)(new_occupations, index_dn - Ns, index_dn)

        log_prob_correction = 0.0

        return state.replace(occupations=new_occupations), allowed_move, log_prob_correction
