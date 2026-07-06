from flax import struct
import jax
import jax.numpy as jnp

from tachys.lattice.state import State
from tachys.lattice.spins.spin_state import SpinState
from tachys.lattice.fermions.fermion_state import FermionState

class FoundationState(struct.PyTreeNode):
    """Per-sample bookkeeping for training one ansatz across many systems at once.

    A foundation model shares a single network across a batch that mixes
    samples drawn from several distinct physical systems (e.g. different
    Hamiltonians or couplings). FoundationState carries the extra per-sample
    metadata this requires. It is meant to be combined via multiple
    inheritance with a physical State subclass rather than used on its own —
    see SpinFoundationState and FermionFoundationState.

    Attributes:
        system_couplings: Hamiltonian couplings (e.g. J, h) of each sample's
            system, shape (N_mc, n_couplings) — one row per sample, matching
            the leading batch dimension of the paired State's config (e.g.
            spins/occupations), not (n_systems, n_couplings).
        system_ids:       Per-sample integer label in [0, n_systems)
            identifying which system each sample belongs to, shape (N_mc,).
        n_systems:        Total number of distinct systems in the batch
            (static; not a pytree leaf).
    """
    system_couplings: jnp.array
    system_ids: jnp.array
    n_systems: int = struct.field(pytree_node=False)


class SpinFoundationState(SpinState, FoundationState):
    """SpinState samples tagged with their originating system for foundation-model training."""


class FermionFoundationState(FermionState, FoundationState):
    """FermionState samples tagged with their originating system for foundation-model training."""