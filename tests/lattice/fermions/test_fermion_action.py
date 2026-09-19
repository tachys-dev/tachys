import jax
import jax.numpy as jnp
import numpy as np
import pytest

from testing_configs import frozen_config
from tachys.lattice.fermions.fermion_action import FermionSpinExchange
from tachys.lattice.fermions.fermion_state import FermionState
from tachys.lattice.lattice_database import square
from tachys.montecarlo import sample
from tachys.utils import same_treedef_and_avals
from tachys.wavefunction import WaveFunction

L = 4
Ns = L * L
N_mc = 32
Ne = 8


@pytest.fixture(scope="module")
def lat():
    return square(shape=(L, L))


@pytest.fixture(scope="module")
def fermion_setup(lat):
    config = frozen_config("square16_ne8_nmc32")
    state = FermionState(occupations=config, lattice=lat, Ne=Ne, Nbands=2)
    action = FermionSpinExchange.create(lat, max_dist=1)
    mc_keys = jax.random.split(jax.random.key(2), N_mc)
    return state, action, mc_keys


# ─── create() ──────────────────────────────────────────────────────────────

def test_create_bonds_shape(lat):
    action = FermionSpinExchange.create(lat, max_dist=1)
    expected = sum(len(src) for _, src, dst in lat.shells(1))
    assert np.asarray(action.bonds).shape == (expected, 2)


def test_create_more_bonds_with_larger_max_dist(lat):
    action1 = FermionSpinExchange.create(lat, max_dist=1)
    action2 = FermionSpinExchange.create(lat, max_dist=2)
    assert len(action2.bonds) > len(action1.bonds)


# ─── __call__ invariants ────────────────────────────────────────────────────

def test_call_output_shapes(fermion_setup):
    state, action, mc_keys = fermion_setup
    new_state, allowed_move, log_prob_correction, _ = action(mc_keys, state)
    assert new_state.occupations.shape == state.occupations.shape
    assert allowed_move.shape == (N_mc,)
    assert log_prob_correction.shape == (N_mc,)


def test_total_occupation_conserved(fermion_setup):
    state, action, mc_keys = fermion_setup
    new_state, _, _, _ = action(mc_keys, state)
    assert jnp.array_equal(state.occupations.sum(axis=-1), new_state.occupations.sum(axis=-1))


def test_per_band_counts_conserved_when_allowed(fermion_setup):
    state, action, mc_keys = fermion_setup
    new_state, allowed_move, _, _ = action(mc_keys, state)

    before = state.occupations.reshape(N_mc, 2, Ns).sum(axis=-1)
    after = new_state.occupations.reshape(N_mc, 2, Ns).sum(axis=-1)

    assert jnp.all(before[allowed_move] == after[allowed_move])
    # sanity: the fixture should exercise both accepted and rejected proposals
    assert jnp.any(allowed_move)
    assert jnp.any(~allowed_move)


def test_state_structure_preserved(fermion_setup):
    state, action, mc_keys = fermion_setup
    new_state, _, _, _ = action(mc_keys, state)
    assert same_treedef_and_avals(state, new_state)


# ─── deterministic exact cases ──────────────────────────────────────────────

def test_two_site_deterministic_flip():
    lat2 = square(shape=(2, 2))
    # up electron at site 0, down electron at site 3; both destination slots empty
    occ = jnp.array([[1, 0, 0, 0, 0, 0, 0, 1]] * N_mc, dtype=jnp.int32)
    state = FermionState(occupations=occ, lattice=lat2, Ne=2, Nbands=2)
    # single candidate bond (0, 3); the random up/down orientation coin flip
    # means only the (up=0, down=3) assignment is physically valid here, so
    # this also exercises the empty-slot/occupied-slot rejection branch
    action = FermionSpinExchange(max_dist=1, bonds=((0, 3),))

    mc_keys = jax.random.split(jax.random.key(0), N_mc)
    new_state, allowed_move, log_prob_correction, _ = action(mc_keys, state)

    assert jnp.any(allowed_move)
    expected = jnp.array([0, 0, 0, 1, 1, 0, 0, 0], dtype=jnp.int32)
    assert jnp.all(new_state.occupations[allowed_move] == expected)
    assert jnp.allclose(log_prob_correction, 0.0)


def test_two_site_deterministic_reject_on_double_occupancy():
    lat2 = square(shape=(2, 2))
    # both up and down electrons at site 0; site 3 fully empty -> neither
    # orientation of the (0, 3) bond has an occupied source at the empty side
    occ = jnp.array([[1, 0, 0, 0, 1, 0, 0, 0]] * N_mc, dtype=jnp.int32)
    state = FermionState(occupations=occ, lattice=lat2, Ne=2, Nbands=2)
    action = FermionSpinExchange(max_dist=1, bonds=((0, 3),))

    mc_keys = jax.random.split(jax.random.key(0), N_mc)
    _, allowed_move, _, _ = action(mc_keys, state)

    assert jnp.all(~allowed_move)


# ─── sample() integration ───────────────────────────────────────────────────

def test_sample_fermion_conserves_occupations(fermion_setup):
    state, action, mc_keys = fermion_setup

    def apply_fn(params, state):
        return jnp.zeros(state.occupations.shape[0])

    wf = WaveFunction(params={}, apply_fn=apply_fn)

    out_state, log_amps, acceptance = sample(5, state, action, mc_keys, wf)

    assert log_amps.shape == (N_mc,)
    before = state.occupations.reshape(N_mc, 2, Ns).sum(axis=-1)
    after = out_state.occupations.reshape(N_mc, 2, Ns).sum(axis=-1)
    assert jnp.array_equal(before, after)
