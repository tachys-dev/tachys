import dataclasses
import warnings

import jax
from flax import struct

from tachys.lattice.lattice import Lattice

class State(struct.PyTreeNode):
    lattice: Lattice = struct.field(pytree_node=False, default=None, kw_only=True)
    N_mc: int        = struct.field(pytree_node=False, default=None, kw_only=True)

    def __post_init__(self):
        if self.N_mc is None:
            object.__setattr__(self, 'N_mc', self.config.shape[0])

        if self.lattice is not None:
            return
        # Skip while retraced inside jit/vmap/scan (dynamic fields are Tracers then) —
        # only warn on genuine, concrete construction.
        for f in dataclasses.fields(self):
            if not f.metadata.get('pytree_node', True):
                continue
            if isinstance(getattr(self, f.name), jax.core.Tracer):
                return
        warnings.warn(f"{type(self).__name__} created without a lattice; "
                      "Ns and other lattice-dependent operations will be unavailable.")

    @property
    def Ns(self):
        return self.lattice.Ns

    @property
    def config(self):
        """The per-chain array that generic lattice moves (e.g. BondExchange) act on."""
        raise NotImplementedError

    def replace_config(self, new_config):
        raise NotImplementedError