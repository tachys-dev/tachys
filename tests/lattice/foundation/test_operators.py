"""Tests for tachys.lattice.foundation.operators (broadcast_coupling,
concatenate_couplings, combine_systems).

Builds one Heisenberg Hamiltonian per system (same L, different J), combines
them into a single foundation-model operator, and checks that applying the
combined operator to a batch mixing samples from every system reproduces
exactly what applying each system's own operator to its own slice would give.
"""
import jax
import jax.numpy as jnp
import pytest

from tachys.lattice.foundation.operators import (
    broadcast_coupling, concatenate_couplings, combine_systems, extract_system_couplings,
)
from tachys.lattice.operator.base import DiagOffdiagResult
from tachys.lattice.spins.hamiltonians.heisenberg import heisenberg_square_pbc
from tachys.lattice.spins.spin_state import SpinState
from tachys.lattice.lattice_database import square
from tachys.lattice.fermions.fermion_operators import Cup, Cup_dag, Cdn, Cdn_dag, Nup, Ndn
from tachys.lattice.fermions.hamiltonians.hubbard import hubbard_hamiltonian

L = 2
N = L * L
Js = [1.0, 2.0]
N_MC_PER_SYSTEM = 3


def _make_state(key, n_mc):
    spins = jax.random.choice(key, jnp.array([-1.0, 1.0]), shape=(n_mc, N))
    return SpinState(spins=spins, lattice=square(shape=(L, L)))


@pytest.fixture(scope="module")
def systems():
    return [heisenberg_square_pbc(L, J=J) for J in Js]


def test_broadcast_coupling_shape(systems):
    H = systems[0]
    n_terms = H.operators[0].operators[0].coupling.shape[0]

    broadcasted = broadcast_coupling(H, N_MC_PER_SYSTEM)
    leaf = broadcasted.operators[0].operators[0]
    assert leaf.coupling.shape == (n_terms, N_MC_PER_SYSTEM)
    # the broadcast axis is a repeat of the same per-term value
    assert jnp.all(leaf.coupling == leaf.coupling[:, :1])


def test_concatenate_couplings_rejects_mismatched_structure(systems):
    broadcasted = [broadcast_coupling(H, N_MC_PER_SYSTEM) for H in systems]
    mismatched = broadcasted[1].operators[0]  # a bare _OperatorMul, not an _OperatorSum
    with pytest.raises(ValueError):
        concatenate_couplings([broadcasted[0], mismatched])


def test_combine_systems_coupling_shape(systems):
    H_comb = combine_systems(systems, N_MC_PER_SYSTEM)
    n_terms = systems[0].operators[0].operators[0].coupling.shape[0]
    n_mc_total = len(systems) * N_MC_PER_SYSTEM

    leaf = H_comb.operators[0].operators[0]
    assert leaf.coupling.shape == (n_terms, n_mc_total)


def test_combine_systems_matches_per_system_application(systems):
    H_comb = combine_systems(systems, N_MC_PER_SYSTEM)

    key = jax.random.key(0)
    keys = jax.random.split(key, len(systems))
    states = [_make_state(k, N_MC_PER_SYSTEM) for k in keys]
    state_comb = states[0].replace(
        spins=jnp.concatenate([s.spins for s in states], axis=0)
    )

    result_comb = H_comb(state_comb)
    assert isinstance(result_comb, DiagOffdiagResult)

    for i, (H_i, state_i) in enumerate(zip(systems, states)):
        sl = slice(i * N_MC_PER_SYSTEM, (i + 1) * N_MC_PER_SYSTEM)
        result_i = H_i(state_i)

        assert jnp.allclose(
            result_comb.diagonal.matrix_element[:, sl], result_i.diagonal.matrix_element
        )
        assert jnp.allclose(
            result_comb.offdiagonal.matrix_element[:, sl], result_i.offdiagonal.matrix_element
        )
        assert jnp.array_equal(
            result_comb.offdiagonal.mask[:, sl], result_i.offdiagonal.mask
        )
        assert jnp.array_equal(
            result_comb.offdiagonal.connected_states.spins[:, sl, :],
            result_i.offdiagonal.connected_states.spins,
        )


def _hubbard_square_pbc(L, t=1.0, U=0.0):
    """Local copy of main_foundation.py's Hubbard builder -- hopping is
    written as four separate Cdag*C leaves (not the generic HoppingUp/
    HoppingDown leaf from hubbard_hamiltonian), so combining systems
    produces genuine cross-leaf redundancy: one U-carrying leaf (Nup)
    alongside four -t leaves with identical, batch-constant values.
    """
    H = None
    for x in range(L):
        for y in range(L):
            i = x * L + y
            interaction = U * Nup(i) * Ndn(i)
            H = interaction if H is None else H + interaction
            for j in (x * L + (y + 1) % L, ((x + 1) % L) * L + y):
                hopping = (  - t * Cup_dag(i) * Cup(j)
                           + - t * Cup_dag(j) * Cup(i)
                           + - t * Cdn_dag(i) * Cdn(j)
                           + - t * Cdn_dag(j) * Cdn(i))
                H = H + hopping
    return H


HUBBARD_L = 2
HUBBARD_US = [0.0, 2.0, 4.0, 8.0]


@pytest.fixture(scope="module")
def hubbard_systems():
    return [_hubbard_square_pbc(HUBBARD_L, U=U) for U in HUBBARD_US]


def test_extract_system_couplings_drops_redundant_hopping_leaves(hubbard_systems):
    H_comb = combine_systems(hubbard_systems, N_MC_PER_SYSTEM)
    couplings = extract_system_couplings(H_comb)

    n_mc = len(HUBBARD_US) * N_MC_PER_SYSTEM
    assert couplings.shape == (n_mc, 1)

    expected = jnp.repeat(jnp.array(HUBBARD_US), N_MC_PER_SYSTEM)
    assert jnp.allclose(couplings[:, 0], expected)


@pytest.fixture(scope="module")
def two_shell_hopping_systems():
    lat = square(shape=(4, 4))
    pairs = [(1.0, 0.3), (2.0, 0.7), (0.5, 0.9)]  # (t1, t2) per system
    return [
        hubbard_hamiltonian(lat, nn=[
            ((1, 0), t1), ((0, 1), t1),
            ((1, 1), t2), ((1, -1), t2),
        ], U=1.0)
        for t1, t2 in pairs
    ]


def test_extract_system_couplings_splits_intra_leaf_shells(two_shell_hopping_systems):
    H_comb = combine_systems(two_shell_hopping_systems, N_MC_PER_SYSTEM)
    couplings = extract_system_couplings(H_comb)

    n_mc = len(two_shell_hopping_systems) * N_MC_PER_SYSTEM
    assert couplings.shape == (n_mc, 2)

    t1s = jnp.repeat(jnp.array([1.0, 2.0, 0.5]), N_MC_PER_SYSTEM)
    t2s = jnp.repeat(jnp.array([0.3, 0.7, 0.9]), N_MC_PER_SYSTEM)
    # HoppingUp's shell-1 rows appear before its shell-2 rows in nn order;
    # HoppingDown duplicates both and drops out in dedup.
    assert jnp.allclose(couplings[:, 0], -t1s)
    assert jnp.allclose(couplings[:, 1], -t2s)


def test_extract_system_couplings_no_variation_with_single_system(systems):
    H_comb = combine_systems(systems[:1], N_MC_PER_SYSTEM)
    couplings = extract_system_couplings(H_comb)
    assert couplings.shape == (N_MC_PER_SYSTEM, 0)
