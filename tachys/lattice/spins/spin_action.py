from functools import partial

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


class SpinExchange(_BaseAction):
    """Proposes swapping two spins separated by at most `max_dist` bonds.

    The neighbour to swap with is drawn uniformly from all neighbours within
    the requested distance, so the proposal is symmetric and
    log_prob_correction = 0.

    Attributes:
        max_dist: Maximum bond distance between the two sites to exchange.
    """

    max_dist: int = struct.field(pytree_node=False)

    def __call__(self, key, state):
        rands = jax.vmap(partial(jax.random.uniform, shape=(3,)))(key)

        n_mult = jnp.array(state.n_mult)
        neighbours = jnp.array(state.neighbours)
        N_mc = state.spins.shape[0]

        index = (rands[:, 0] * state.Ns).astype(int)
        dist_neigh = (rands[:, 1] * self.max_dist + 1).astype(int)
        pos_neigh = (rands[:, 2] * n_mult[dist_neigh]).astype(int)
        index_nn = neighbours[index, pos_neigh, dist_neigh]

        allowed_move = state.spins[jnp.arange(N_mc), index] != state.spins[jnp.arange(N_mc), index_nn]
        new_spins = jax.vmap(exchange_spins)(state.spins, index, index_nn)
        log_prob_correction = 0.0

        return state.replace(spins=new_spins), allowed_move, log_prob_correction


class SpinFlip(_BaseAction):
    """Proposes flipping a randomly chosen spin.

    The site is drawn uniformly, so the proposal is symmetric and
    log_prob_correction = 0.
    """

    def __call__(self, key, state):
        rands = jax.vmap(jax.random.uniform)(key)
        index = (rands * state.Ns).astype(int)
        N_mc = state.spins.shape[0]

        new_spins = state.spins.at[jnp.arange(N_mc), index].set(-state.spins[jnp.arange(N_mc), index])
        allowed_move = new_spins[..., 0] != 0  # always true
        log_prob_correction = 0.0

        return state.replace(spins=new_spins), allowed_move, log_prob_correction
