import jax

from tachys.parallel import n_devices
from tachys.lattice.spins.spin_state import SpinState
from tachys.lattice.fermions.fermion_state import FermionState


def get_n_mc_local(state):
    """The batch size of state along axis 0 as it currently stands (N_mc_local
    if state is a per-device shard, else the same as get_n_mc) — read from any
    one data leaf, since every State subclass's data fields share the same
    leading batch axis (State.__post_init__ checks this)."""
    return jax.tree.leaves(state)[0].shape[0]


def get_n_mc(state):
    """The global Monte Carlo batch size, even when state is currently a
    per-device shard inside a shard_map body."""
    return get_n_mc_local(state) * n_devices


def get_array(state):
    """The per-walker physical array a State subclass wraps (spins/occupations)."""
    if isinstance(state, SpinState):
        return state.spins
    if isinstance(state, FermionState):
        return state.occupations
    raise TypeError(f"get_array: unsupported State subclass {type(state).__name__}")


def replace_array(state, new_array):
    """Return a copy of state with its physical array replaced by new_array."""
    if isinstance(state, SpinState):
        return state.replace(spins=new_array)
    if isinstance(state, FermionState):
        return state.replace(occupations=new_array)
    raise TypeError(f"replace_array: unsupported State subclass {type(state).__name__}")
