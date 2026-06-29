from flax import struct
import jax.numpy as jnp

class State(struct.PyTreeNode):
    Ns: int = struct.field(pytree_node=False)