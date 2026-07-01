from flax import struct
import jax.numpy as jnp

class State(struct.PyTreeNode):
    Ns: int = struct.field(pytree_node=False)

    @property
    def config(self):
        """The per-chain array that generic lattice moves (e.g. BondExchange) act on."""
        raise NotImplementedError

    def replace_config(self, new_config):
        raise NotImplementedError