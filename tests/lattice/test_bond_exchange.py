import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tachys.lattice.ansatz.rbm import SpinRBM
from tachys.lattice.bond_exchange import BondExchange
from tachys.lattice.fermions.fermion_state import FermionState, init_config_spinful
from tachys.lattice.lattice_database import chain, square
from tachys.lattice.spins.spin_action import SpinFlip
from tachys.lattice.spins.spin_state import SpinState, init_config_fixed_magn
from tachys.montecarlo import CompositeAction, sample
from tachys.utils import same_treedef_and_avals
from tachys.wavefunction import WaveFunction

L = 4
Ns = L * L
N_mc = 16
Ne = 4


@pytest.fixture(scope="module")
def lat():
    return square(shape=(L, L))


@pytest.fixture(scope="module")
def fermion_setup(lat):
    config, _, _ = init_config_spinful(jax.random.key(1), Ns, Ne=Ne, N_mc=N_mc)
    state = FermionState(occupations=config, lattice=lat, Ne=Ne, Nbands=2)
    action = BondExchange.create(lat, max_dist=1, Nbands=2)
    mc_keys = jax.random.split(jax.random.key(2), N_mc)
    return state, action, mc_keys


@pytest.fixture(scope="module")
def spin_setup(lat):
    spins = init_config_fixed_magn(jax.random.key(1), Ns, sz=0, N_mc=N_mc)
    state = SpinState(spins=spins, lattice=lat)
    action = BondExchange.create(lat, max_dist=1)
    mc_keys = jax.random.split(jax.random.key(2), N_mc)
    return state, action, mc_keys


# ─── create() ──────────────────────────────────────────────────────────────

def test_create_bonds_shape(lat):
    action = BondExchange.create(lat, max_dist=1)
    expected = sum(len(src) for _, src, dst in lat.shells(1))
    assert np.asarray(action.bonds).shape == (expected, 2)


def test_create_more_bonds_with_larger_max_dist(lat):
    action1 = BondExchange.create(lat, max_dist=1)
    action2 = BondExchange.create(lat, max_dist=2)
    assert len(action2.bonds) > len(action1.bonds)


# ─── __call__ invariants (fermion + spin) ───────────────────────────────────

@pytest.mark.parametrize("setup_name,Nbands", [("fermion_setup", 2), ("spin_setup", 1)])
def test_call_output_shapes(setup_name, Nbands, request):
    state, action, mc_keys = request.getfixturevalue(setup_name)
    new_state, allowed_move, log_prob_correction, _ = action(mc_keys, state)
    assert new_state.config.shape == state.config.shape
    assert allowed_move.shape == (N_mc,)
    assert log_prob_correction.shape == (N_mc,)


@pytest.mark.parametrize("setup_name", ["fermion_setup", "spin_setup"])
def test_allowed_move_always_true(setup_name, request):
    state, action, mc_keys = request.getfixturevalue(setup_name)
    _, allowed_move, _, _ = action(mc_keys, state)
    assert jnp.all(allowed_move)


@pytest.mark.parametrize("setup_name", ["fermion_setup", "spin_setup"])
def test_exactly_two_sites_change(setup_name, request):
    state, action, mc_keys = request.getfixturevalue(setup_name)
    new_state, _, _, _ = action(mc_keys, state)
    n_changed = (state.config != new_state.config).sum(axis=-1)
    assert jnp.all(n_changed == 2)


def test_fermion_swap_stays_within_band(fermion_setup):
    state, action, mc_keys = fermion_setup
    new_state, _, _, _ = action(mc_keys, state)
    changed = np.asarray(state.occupations != new_state.occupations)
    for row in changed:
        idx = np.flatnonzero(row)
        assert idx.shape == (2,)
        assert idx[0] // Ns == idx[1] // Ns


@pytest.mark.parametrize("setup_name,Nbands", [("fermion_setup", 2), ("spin_setup", 1)])
def test_per_band_counts_conserved(setup_name, Nbands, request):
    state, action, mc_keys = request.getfixturevalue(setup_name)
    new_state, _, _, _ = action(mc_keys, state)
    before = state.config.reshape(N_mc, Nbands, Ns).sum(axis=-1)
    after = new_state.config.reshape(N_mc, Nbands, Ns).sum(axis=-1)
    assert jnp.array_equal(before, after)


@pytest.mark.parametrize("setup_name", ["fermion_setup", "spin_setup"])
def test_state_structure_preserved(setup_name, request):
    state, action, mc_keys = request.getfixturevalue(setup_name)
    new_state, _, _, _ = action(mc_keys, state)
    assert same_treedef_and_avals(state, new_state)


# ─── deterministic exact-swap case ──────────────────────────────────────────

def test_two_site_chain_deterministic_swap():
    lat2 = chain(2, pbc=False)
    spins = jnp.array([[-1, 1]] * N_mc, dtype=jnp.int8)
    state = SpinState(spins=spins, lattice=lat2)
    action = BondExchange.create(lat2, max_dist=1)

    mc_keys = jax.random.split(jax.random.key(0), N_mc)
    new_state, allowed_move, log_prob_correction, _ = action(mc_keys, state)

    assert jnp.all(allowed_move)
    assert jnp.array_equal(new_state.spins, jnp.array([[1, -1]] * N_mc, dtype=jnp.int8))
    # only one candidate bond, valid both before and after -> no correction
    assert jnp.allclose(log_prob_correction, 0.0)


# ─── sample() integration ───────────────────────────────────────────────────

def test_sample_spin_conserves_magnetization(spin_setup):
    state, action, mc_keys = spin_setup

    model = SpinRBM(hidden_units=1, dtype=jnp.float64, complex=True)
    dummy = SpinState(spins=jnp.ones((1, Ns), dtype=jnp.float64), lattice=state.lattice)
    params = model.init(jax.random.key(0), dummy)
    wf = WaveFunction(params=params, apply_fn=model.apply)

    out_state, log_amps, acceptance = sample(5, state, action, mc_keys, wf)

    assert log_amps.shape == (N_mc,)
    assert jnp.all(acceptance >= 0.0) and jnp.all(acceptance <= 1.0)
    assert jnp.array_equal(state.spins.sum(axis=-1), out_state.spins.sum(axis=-1))


def test_sample_fermion_conserves_occupations(fermion_setup):
    state, action, mc_keys = fermion_setup

    def apply_fn(params, state):
        return jnp.zeros(state.config.shape[0])

    wf = WaveFunction(params={}, apply_fn=apply_fn)

    out_state, log_amps, acceptance = sample(5, state, action, mc_keys, wf)

    assert log_amps.shape == (N_mc,)
    before = state.occupations.reshape(N_mc, 2, Ns).sum(axis=-1)
    after = out_state.occupations.reshape(N_mc, 2, Ns).sum(axis=-1)
    assert jnp.array_equal(before, after)


# ─── CompositeAction integration ────────────────────────────────────────────
# Regression coverage: combining BondExchange with another action inside
# CompositeAction used to fail two ways.
#   1. BondExchange.bonds was an unhashable jnp.array marked static, but
#      CompositeAction dispatches sub-actions via jax.lax.switch, which
#      requires every branch (each a pytree with static/aux data) to be
#      hashable.
#   2. Even once hashable, jax.lax.switch also requires every branch's
#      *output* shapes to match exactly. BondExchange's log_prob_correction
#      is a genuine per-chain array (it depends on how many valid bonds exist
#      before/after the move), while symmetric moves like SpinFlip return a
#      bare 0.0 constant — these didn't have matching shapes.

@pytest.fixture(scope="module")
def composite_setup(lat):
    spins = init_config_fixed_magn(jax.random.key(1), Ns, sz=0, N_mc=N_mc)
    state = SpinState(spins=spins, lattice=lat)
    action = CompositeAction(
        actions=(BondExchange.create(lat, max_dist=1), SpinFlip()),
        probs=(0.5, 0.5),
    )
    mc_keys = jax.random.split(jax.random.key(2), N_mc)
    return state, action, mc_keys


def test_composite_bond_exchange_runs(composite_setup):
    """The real regression: this used to raise before the switch was fixed."""
    state, action, mc_keys = composite_setup
    new_state, allowed_move, log_prob_correction, action_id = action(mc_keys, state)

    assert new_state.spins.shape == state.spins.shape
    assert allowed_move.shape == (N_mc,)
    assert log_prob_correction.shape == (N_mc,)
    assert action_id.shape == (N_mc,)


def test_composite_bond_exchange_selects_both_actions(composite_setup):
    state, action, mc_keys = composite_setup
    _, _, _, action_id = action(mc_keys, state)
    assert set(np.unique(np.asarray(action_id)).tolist()) == {0, 1}


def test_composite_bond_exchange_dtypes_preserved(composite_setup):
    state, action, mc_keys = composite_setup
    new_state, allowed_move, _, _ = action(mc_keys, state)
    assert new_state.spins.dtype == state.spins.dtype
    assert allowed_move.dtype == jnp.bool_


def test_composite_bond_exchange_move_matches_selected_action(composite_setup):
    """BondExchange always swaps exactly 2 sites; SpinFlip always flips exactly 1."""
    state, action, mc_keys = composite_setup
    new_state, _, _, action_id = action(mc_keys, state)

    n_changed = np.asarray(state.spins != new_state.spins).sum(axis=-1)
    action_id = np.asarray(action_id)
    assert np.all(n_changed[action_id == 0] == 2)
    assert np.all(n_changed[action_id == 1] == 1)


def test_composite_bond_exchange_sample_runs(composite_setup):
    """The same combination, driven through the full sample() Metropolis loop."""
    state, action, mc_keys = composite_setup

    model = SpinRBM(hidden_units=1, dtype=jnp.float64, complex=True)
    dummy = SpinState(spins=jnp.ones((1, Ns), dtype=jnp.float64), lattice=state.lattice)
    params = model.init(jax.random.key(0), dummy)
    wf = WaveFunction(params=params, apply_fn=model.apply)

    out_state, log_amps, acceptance = sample(5, state, action, mc_keys, wf)

    assert log_amps.shape == (N_mc,)
    assert jnp.all(acceptance >= 0.0) and jnp.all(acceptance <= 1.0)
    assert out_state.spins.shape == state.spins.shape
