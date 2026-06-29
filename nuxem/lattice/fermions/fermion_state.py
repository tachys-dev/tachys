from flax import struct
import jax
import jax.numpy as jnp

from nuxem.lattice.state import State

class FermionState(State):
    occupations: jnp.array
    Ne: int = struct.field(pytree_node=False)
    Nbands: int = struct.field(pytree_node=False, default=2) #* single-band spinful fermions by default

def init_config_spinful(key, Ns, Ne, sz=0, N_mc=1, particle_hole=False):
    assert Ne % 2 == 0

    N_up = Ne // 2 + sz
    N_down = Ne // 2 - sz

    if particle_hole:
        N_down = Ns - N_down
    
    config = jnp.zeros((N_mc, 2*Ns), dtype=jnp.int32)
    config = config.at[:, :N_up].set(1)
    config = config.at[:, Ns:Ns+N_down].set(1)
    
    subkey1, subkey2 = jax.random.split(key, num=2)
    config1 = jax.random.permutation(subkey1, config[:, :Ns], axis=1, independent=True)
    config2 = jax.random.permutation(subkey2, config[:, Ns:], axis=1, independent=True)
    config = jnp.concatenate((config1, config2), axis=-1)

    return config, N_up, N_down