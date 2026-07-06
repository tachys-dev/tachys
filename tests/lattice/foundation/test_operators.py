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
    broadcast_coupling, concatenate_couplings, combine_systems,
)
from tachys.lattice.operator.base import DiagOffdiagResult
from tachys.lattice.spins.hamiltonians.heisenberg import heisenberg_square_pbc
from tachys.lattice.spins.spin_state import SpinState
from tachys.lattice.lattice_database import square

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
