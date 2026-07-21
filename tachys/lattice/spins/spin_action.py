import jax
import jax.numpy as jnp

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
