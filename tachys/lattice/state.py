import dataclasses
import warnings

import jax
from flax import struct

from tachys.lattice.lattice import Lattice

class State(struct.PyTreeNode):
    lattice: Lattice = struct.field(pytree_node=False, default=None, kw_only=True)

    def __post_init__(self):
        # Every data leaf's leading axis is the MC-batch dimension (N_mc, or
        # N_mc_local under sharding) — get_n_mc_local/get_n_mc rely on this
        # holding for any one leaf, so check it holds for all of them. Shapes
        # are always concrete even under jit/vmap tracing, so this is safe to
        # run unconditionally; skip leaves with no real shape info (e.g. bare
        # placeholder sentinels, or orbax's ArrayRestoreArgs with shape=None)
        # that shard_map/orbax substitute in during structure-only pytree
        # reconstruction, never seen by user code. Also skip leaves with fewer
        # than 2 dims: e.g. jax.vmap over a FoundationState's batch axis (as
        # in _kernels.py's per-sample Jacobian) legitimately leaves each field
        # with its own, individually-meaningful remaining axis (site count,
        # coupling count, ...) once the shared batch axis is stripped — those
        # aren't supposed to agree, only genuinely-still-batched (>=2D) leaves
        # (occupations/system_couplings) are.
        shapes = (getattr(leaf, 'shape', None) for leaf in jax.tree.leaves(self))
        sizes = {shape[0] for shape in shapes if shape and len(shape) >= 2}
        if len(sizes) > 1:
            raise ValueError(
                f"{type(self).__name__}: all data leaves must share the same "
                f"leading (N_mc) axis size, got {sorted(sizes)}"
            )

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