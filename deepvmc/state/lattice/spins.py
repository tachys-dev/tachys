from flax import struct
import jax.numpy as jnp

class SpinState(struct.PyTreeNode):
    spins: jnp.array
    Ns: int