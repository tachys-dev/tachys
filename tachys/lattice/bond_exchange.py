import numpy as np
import jax
import jax.numpy as jnp
from flax import struct

from tachys.montecarlo import _BaseAction
from tachys.lattice.spins.spin_action import exchange_spins


def _compute_valid_bonds_mask(bonds, config):
    """Mask of bonds whose two sites hold different values (a valid exchange)."""
    return config[bonds[:, 0]] != config[bonds[:, 1]]


class BondExchange(_BaseAction):
    """Proposes exchanging two sites' values within the same band.

    Candidate site pairs are precomputed from the lattice geometry (all bonds up
    to `max_dist` shells). On each step, a band is drawn uniformly, then a pair is
    drawn uniformly among the bonds (within that band) that are currently valid
    (the two sites differ). Because the proposal is restricted to the valid
    subset, and the count of valid bonds generally differs before/after the move,
    the proposal is asymmetric and needs a log-probability correction.

    Works with any State subclass that implements `.config` / `.replace_config`
    (e.g. SpinState.spins or FermionState.occupations).

    Attributes:
        max_dist: Maximum bond distance (in lattice shells) between the two sites.
        bonds: [N_bonds, 2] int array of candidate (site_i, site_j) pairs.
        Nbands: Number of bands sharing the same site indexing (e.g. 2 for
            spin-1/2 fermion occupations, 1 for plain spins).
    """

    max_dist: int = struct.field(pytree_node=False)
    bonds: tuple = struct.field(pytree_node=False)
    Nbands: int = struct.field(pytree_node=False, default=1)

    @classmethod
    def create(cls, lattice, max_dist=1, Nbands=1):
        bonds = [np.column_stack((src, dst)) for _, src, dst in lattice.shells(max_dist)]
        bonds = np.vstack(bonds)

        # A hashable tuple-of-tuples, not a jnp.array, so BondExchange instances stay
        # hashable — required for jax.lax.switch when composed inside CompositeAction.
        return cls(max_dist=max_dist, bonds=tuple(map(tuple, bonds.tolist())), Nbands=Nbands)

    def __call__(self, key, state):
        key = jax.vmap(jax.random.split)(key)
        subkey1, subkey2 = key[:, 0], key[:, 1]

        Ns = state.Ns
        bonds_arr = jnp.asarray(self.bonds)
        N_bonds = bonds_arr.shape[0]
        N_mc = state.config.shape[0]

        rands = jax.vmap(jax.random.uniform)(subkey1)
        band_index = (rands * self.Nbands).astype(int)
        #* notice no inter-band mixing
        bonds = band_index[:, None, None] * Ns + bonds_arr  # [N_mc, N_bonds, 2]

        valid_mask = jax.vmap(_compute_valid_bonds_mask)(bonds, state.config)
        sampled_indices = jax.vmap(lambda p, k: jax.random.choice(k, N_bonds, p=p))(valid_mask, subkey2)

        i = bonds[jnp.arange(N_mc), sampled_indices, 0]
        j = bonds[jnp.arange(N_mc), sampled_indices, 1]

        new_config = jax.vmap(exchange_spins)(state.config, i, j)
        new_valid_mask = jax.vmap(_compute_valid_bonds_mask)(bonds, new_config)

        log_prob_correction = jnp.log(valid_mask.sum(-1)) - jnp.log(new_valid_mask.sum(-1))
        allowed_move = i != j  # always true by construction

        return state.replace_config(new_config), allowed_move, log_prob_correction
