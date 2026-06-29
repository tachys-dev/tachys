from flax import struct
import jax
import jax.numpy as jnp

from nuxem.lattice.state import State

class SpinState(State):
    spins: jnp.array

def init_config_fixed_magn(key, N, sz=0, N_mc=1):
    N_m = N // 2 - sz
    config = jnp.ones((N_mc, N), dtype=jnp.int8)
    config = config.at[:, :N_m].set(-1)

    config = jax.random.permutation(key, config, axis=1, independent=True)

    assert jnp.sum(config) / (N_mc) == 2*sz

    return config