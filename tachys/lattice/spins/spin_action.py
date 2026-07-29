import numpy as np
import jax
import jax.numpy as jnp
from flax import struct

from tachys.montecarlo import _BaseAction


def exchange_spins(spins, id1, id2):
    """Swap spins at positions id1 and id2 in a single-chain spin array."""
    aus = spins[id1]
    spins = spins.at[id1].set(spins[id2])
    spins = spins.at[id2].set(aus)
    return spins


class SpinFlip(_BaseAction):
    """Proposes flipping a randomly chosen spin.

    The site is drawn uniformly, so the proposal is symmetric and
    log_prob_correction = 0.
    """

    def __call__(self, key, state):
        rands = jax.vmap(jax.random.uniform)(key)
        index = (rands * state.Ns).astype(int)
        N_mc_local = state.spins.shape[0]

        new_spins = state.spins.at[jnp.arange(N_mc_local), index].set(-state.spins[jnp.arange(N_mc_local), index])
        allowed_move = new_spins[..., 0] != 0  # always true
        log_prob_correction = 0.0

        return state.replace(spins=new_spins), allowed_move, log_prob_correction


class BondFlip(_BaseAction):
    """Proposes flipping BOTH spins on a randomly chosen bond, unconditionally.

    Mirrors the process an off-diagonal bond term built from two
    unconditional single-site flip operators (e.g. spin_operators.Sx/Sy)
    connects to: both endpoint spins flip, regardless of their current
    values -- unlike BondExchange (swap, only valid when the two spins
    differ, Sz-conserving) or SpinFlip (single random site, bond-agnostic).

    The bond is drawn uniformly from a fixed precomputed list, independent
    of the current configuration, so the proposal is its own inverse and
    log_prob_correction = 0.

    Attributes:
        bonds: [N_bonds, 2] tuple of candidate (site_i, site_j) pairs.
    """

    bonds: tuple = struct.field(pytree_node=False)

    @classmethod
    def create(cls, lattice, deltas, b_from=0, b_to=None):
        """Pools one or more cell displacements into a single candidate bond list.

        Pass multiple `deltas` (e.g. Kitaev's x- and y-bond displacements)
        to pool several bond types into one action.
        """
        bonds = []
        for delta in deltas:
            src, dst = lattice.bonds(delta, b_from=b_from, b_to=b_to)
            bonds.append(np.column_stack((src, dst)))
        bonds = np.vstack(bonds)
        return cls(bonds=tuple(map(tuple, bonds.tolist())))

    def __call__(self, key, state):
        bonds_arr = jnp.asarray(self.bonds)
        N_bonds = bonds_arr.shape[0]
        N_mc_local = state.spins.shape[0]
        chain_idx = jnp.arange(N_mc_local)

        rands = jax.vmap(jax.random.uniform)(key)
        idx = (rands * N_bonds).astype(int)
        i, j = bonds_arr[idx, 0], bonds_arr[idx, 1]

        new_spins = state.spins.at[chain_idx, i].set(-state.spins[chain_idx, i])
        new_spins = new_spins.at[chain_idx, j].set(-new_spins[chain_idx, j])

        allowed_move = jnp.ones(N_mc_local, dtype=bool)
        log_prob_correction = 0.0

        return state.replace(spins=new_spins), allowed_move, log_prob_correction
