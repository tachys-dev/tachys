"""Tests that SpinFoundationState / FermionFoundationState compose their two
parents (SpinState/FermionState and FoundationState) correctly: the MRO
prefers the physical State subclass, the merged dataclass has no field-
ordering conflicts and carries every field from both parents, State behavior
(config, Ns, replace) still resolves correctly, and the result is a proper
JAX pytree with the right static/dynamic split.
"""
import dataclasses

import jax
import jax.numpy as jnp
import pytest

from tachys.lattice.foundation.foundation_state import (
    FoundationState, SpinFoundationState, FermionFoundationState,
)
from tachys.lattice.spins.spin_state import SpinState
from tachys.lattice.fermions.fermion_state import FermionState
from tachys.lattice.state import State
from tachys.lattice.lattice_database import square

N_mc = 3
LATTICE = square(shape=(2, 2))
N = LATTICE.Ns
N_COUPLINGS = 2


def _make_spin_state(n_systems=2):
    return SpinFoundationState(
        spins=jnp.ones((N_mc, N)),
        system_couplings=jnp.arange(N_mc * N_COUPLINGS, dtype=jnp.float64).reshape(N_mc, N_COUPLINGS),
        system_ids=jnp.array([0, 1, 0]),
        n_systems=n_systems,
        lattice=LATTICE,
    )


def _make_fermion_state(n_systems=2):
    return FermionFoundationState(
        occupations=jnp.zeros((N_mc, 2 * N), dtype=jnp.int32),
        Ne=4,
        system_couplings=jnp.arange(N_mc * N_COUPLINGS, dtype=jnp.float64).reshape(N_mc, N_COUPLINGS),
        system_ids=jnp.array([0, 1, 0]),
        n_systems=n_systems,
        lattice=LATTICE,
    )


# ---------------------------------------------------------------------------
# MRO / subclassing
# ---------------------------------------------------------------------------

def test_spin_foundation_state_subclasses_both_parents():
    assert issubclass(SpinFoundationState, SpinState)
    assert issubclass(SpinFoundationState, FoundationState)
    assert issubclass(SpinFoundationState, State)


def test_fermion_foundation_state_subclasses_both_parents():
    assert issubclass(FermionFoundationState, FermionState)
    assert issubclass(FermionFoundationState, FoundationState)
    assert issubclass(FermionFoundationState, State)


def test_spin_foundation_state_mro_prefers_spin_state():
    """SpinState is listed first, so its methods must win over FoundationState's."""
    mro = SpinFoundationState.__mro__
    assert mro.index(SpinState) < mro.index(FoundationState)


def test_fermion_foundation_state_mro_prefers_fermion_state():
    mro = FermionFoundationState.__mro__
    assert mro.index(FermionState) < mro.index(FoundationState)


# ---------------------------------------------------------------------------
# Field composition — this is where a bad diamond (dataclass field ordering,
# a field silently shadowed, etc.) would show up.
# ---------------------------------------------------------------------------

def test_spin_foundation_state_has_fields_from_both_parents():
    names = {f.name for f in dataclasses.fields(SpinFoundationState)}
    assert names == {"spins", "lattice", "system_couplings", "system_ids", "n_systems"}


def test_fermion_foundation_state_has_fields_from_both_parents():
    names = {f.name for f in dataclasses.fields(FermionFoundationState)}
    assert names == {
        "occupations", "Ne", "Nbands", "lattice",
        "system_couplings", "system_ids", "n_systems",
    }


def test_static_vs_pytree_fields_preserved_through_inheritance():
    """lattice and n_systems must stay non-pytree metadata after the merge —
    a broken diamond could silently drop the pytree_node=False annotation."""
    static = {f.name for f in dataclasses.fields(SpinFoundationState)
              if not f.metadata.get("pytree_node", True)}
    assert static == {"lattice", "n_systems"}


def test_fermion_default_field_survives_the_merge():
    """Nbands has a default in FermionState; the merge must not require it
    to be re-specified nor drop the default."""
    f = FermionFoundationState(
        occupations=jnp.zeros((N_mc, 2 * N), dtype=jnp.int32),
        Ne=4,
        system_couplings=jnp.zeros((N_mc, N_COUPLINGS)),
        system_ids=jnp.zeros(N_mc, dtype=jnp.int32),
        n_systems=1,
        lattice=LATTICE,
    )
    assert f.Nbands == 2


# ---------------------------------------------------------------------------
# Instantiation + State behavior delegated to the physical subclass
# ---------------------------------------------------------------------------

def test_spin_foundation_state_config_and_lattice_props():
    s = _make_spin_state()
    assert s.config.shape == (N_mc, N)
    assert jnp.array_equal(s.config, s.spins)
    assert s.Ns == N


def test_fermion_foundation_state_config_and_lattice_props():
    f = _make_fermion_state()
    assert f.config.shape == (N_mc, 2 * N)
    assert jnp.array_equal(f.config, f.occupations)
    assert f.Ns == N


def test_replace_config_updates_only_the_physical_field():
    s = _make_spin_state()
    new_spins = -s.spins
    s2 = s.replace_config(new_spins)
    assert jnp.array_equal(s2.spins, new_spins)
    assert jnp.array_equal(s2.system_ids, s.system_ids)
    assert s2.n_systems == s.n_systems


def test_replace_is_immutable():
    s = _make_spin_state()
    s2 = s.replace(system_ids=jnp.array([1, 1, 1]))
    assert jnp.array_equal(s.system_ids, jnp.array([0, 1, 0]))   # original untouched
    assert jnp.array_equal(s2.system_ids, jnp.array([1, 1, 1]))


# ---------------------------------------------------------------------------
# Pytree correctness: dynamic vs static leaves, jit round-trip
# ---------------------------------------------------------------------------

def test_pytree_leaves_are_exactly_the_dynamic_fields():
    s = _make_spin_state()
    leaves = jax.tree_util.tree_leaves(s)
    assert len(leaves) == 3  # spins, system_couplings, system_ids
    assert sum(l.size for l in leaves) == s.spins.size + s.system_couplings.size + s.system_ids.size


def test_pytree_round_trip_preserves_static_and_dynamic_data():
    s = _make_spin_state()
    leaves, treedef = jax.tree_util.tree_flatten(s)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)

    assert isinstance(rebuilt, SpinFoundationState)
    assert jnp.array_equal(rebuilt.spins, s.spins)
    assert jnp.array_equal(rebuilt.system_couplings, s.system_couplings)
    assert rebuilt.n_systems == s.n_systems
    assert rebuilt.lattice is s.lattice


def test_jit_traces_dynamic_fields_and_bakes_in_static_fields():
    @jax.jit
    def f(state):
        return state.spins.sum() + state.system_couplings.sum(), state.n_systems

    s = _make_spin_state(n_systems=2)
    total, n_systems = f(s)
    assert jnp.isclose(total, s.spins.sum() + s.system_couplings.sum())
    assert n_systems == 2

    s2 = _make_spin_state(n_systems=5)
    assert f(s2)[1] == 5


def test_n_systems_is_static_enough_for_segment_sum_num_segments():
    """If the diamond inheritance silently dropped n_systems' pytree_node=False
    metadata, it would arrive inside jit as a Tracer, and passing a Tracer as
    segment_sum's `num_segments` raises — this is exactly the property
    grouped_sum/grouped_mean rely on (tachys.lattice.foundation.collectives)."""
    s = _make_spin_state(n_systems=2)

    @jax.jit
    def f(state):
        return jax.ops.segment_sum(
            state.system_couplings, state.system_ids, num_segments=state.n_systems)

    out = f(s)
    assert out.shape == (2, N_COUPLINGS)
