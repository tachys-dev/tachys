import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tachys.lattice.ansatz.rbm import SpinRBM
from tachys.lattice.lattice_database import chain, square
from tachys.lattice.spins.spin_action import BondFlip, SpinFlip
from tachys.lattice.spins.spin_state import SpinState, init_config_fixed_magn
from tachys.montecarlo import CompositeAction, sample
from tachys.utils import same_treedef_and_avals
from tachys.wavefunction import WaveFunction

L = 4
Ns = L * L
N_mc = 16


@pytest.fixture(scope="module")
def lat():
    return square(shape=(L, L))


@pytest.fixture(scope="module")
def spin_setup(lat):
    spins = init_config_fixed_magn(jax.random.key(1), Ns, sz=0, N_mc=N_mc)
    state = SpinState(spins=spins, lattice=lat)
    action = BondFlip.create(lat, deltas=[(1, 0), (0, 1)])
    mc_keys = jax.random.split(jax.random.key(2), N_mc)
    return state, action, mc_keys


# ─── create() ───────────────────────────────────────────────────────────────

def test_create_bonds_shape_single_delta(lat):
    action = BondFlip.create(lat, deltas=[(1, 0)])
    expected = len(lat.bonds((1, 0))[0])
    assert np.asarray(action.bonds).shape == (expected, 2)


def test_create_pools_multiple_deltas(lat):
    action_x = BondFlip.create(lat, deltas=[(1, 0)])
    action_xy = BondFlip.create(lat, deltas=[(1, 0), (0, 1)])
    assert len(action_xy.bonds) == 2 * len(action_x.bonds)


# ─── __call__ invariants ────────────────────────────────────────────────────

def test_call_output_shapes(spin_setup):
    state, action, mc_keys = spin_setup
    new_state, allowed_move, log_prob_correction, _ = action(mc_keys, state)
    assert new_state.spins.shape == state.spins.shape
    assert allowed_move.shape == (N_mc,)
    assert log_prob_correction.shape == (N_mc,)


def test_allowed_move_always_true(spin_setup):
    state, action, mc_keys = spin_setup
    _, allowed_move, _, _ = action(mc_keys, state)
    assert jnp.all(allowed_move)


def test_log_prob_correction_always_zero(spin_setup):
    state, action, mc_keys = spin_setup
    _, _, log_prob_correction, _ = action(mc_keys, state)
    assert jnp.allclose(log_prob_correction, 0.0)


def test_exactly_two_sites_change(spin_setup):
    state, action, mc_keys = spin_setup
    new_state, _, _, _ = action(mc_keys, state)
    n_changed = (state.spins != new_state.spins).sum(axis=-1)
    assert jnp.all(n_changed == 2)


def test_state_structure_preserved(spin_setup):
    state, action, mc_keys = spin_setup
    new_state, _, _, _ = action(mc_keys, state)
    assert same_treedef_and_avals(state, new_state)


# ─── deterministic exact-flip case ──────────────────────────────────────────

def test_two_site_chain_deterministic_flip():
    lat2 = chain(2, pbc=False)
    spins = jnp.array([[-1, 1]] * N_mc, dtype=jnp.int8)
    state = SpinState(spins=spins, lattice=lat2)
    action = BondFlip.create(lat2, deltas=[(1, 0)])

    mc_keys = jax.random.split(jax.random.key(0), N_mc)
    new_state, allowed_move, log_prob_correction, _ = action(mc_keys, state)

    assert jnp.all(allowed_move)
    assert jnp.array_equal(new_state.spins, jnp.array([[1, -1]] * N_mc, dtype=jnp.int8))
    assert jnp.allclose(log_prob_correction, 0.0)


def test_two_site_chain_flips_equal_spins():
    """Unlike BondExchange (a no-op swap when both spins already agree),
    BondFlip flips both sites unconditionally."""
    lat2 = chain(2, pbc=False)
    spins = jnp.array([[1, 1]] * N_mc, dtype=jnp.int8)
    state = SpinState(spins=spins, lattice=lat2)
    action = BondFlip.create(lat2, deltas=[(1, 0)])

    mc_keys = jax.random.split(jax.random.key(0), N_mc)
    new_state, allowed_move, log_prob_correction, _ = action(mc_keys, state)

    assert jnp.all(allowed_move)
    assert jnp.array_equal(new_state.spins, jnp.array([[-1, -1]] * N_mc, dtype=jnp.int8))
    assert jnp.allclose(log_prob_correction, 0.0)


# ─── sample() integration ───────────────────────────────────────────────────

def test_sample_spin_runs(spin_setup):
    state, action, mc_keys = spin_setup

    model = SpinRBM(hidden_units=1, dtype=jnp.float64, complex=True)
    dummy = SpinState(spins=jnp.ones((1, Ns), dtype=jnp.float64), lattice=state.lattice)
    params = model.init(jax.random.key(0), dummy)
    wf = WaveFunction(params=params, apply_fn=model.apply)

    out_state, log_amps, acceptance = sample(5, state, action, mc_keys, wf)

    assert log_amps.shape == (N_mc,)
    assert jnp.all(acceptance >= 0.0) and jnp.all(acceptance <= 1.0)
    assert out_state.spins.shape == state.spins.shape


# ─── CompositeAction integration ────────────────────────────────────────────

@pytest.fixture(scope="module")
def composite_setup(lat):
    spins = init_config_fixed_magn(jax.random.key(1), Ns, sz=0, N_mc=N_mc)
    state = SpinState(spins=spins, lattice=lat)
    action = CompositeAction(
        actions=(BondFlip.create(lat, deltas=[(1, 0), (0, 1)]), SpinFlip()),
        probs=(0.5, 0.5),
    )
    mc_keys = jax.random.split(jax.random.key(2), N_mc)
    return state, action, mc_keys


def test_composite_bond_flip_runs(composite_setup):
    state, action, mc_keys = composite_setup
    new_state, allowed_move, log_prob_correction, action_id = action(mc_keys, state)

    assert new_state.spins.shape == state.spins.shape
    assert allowed_move.shape == (N_mc,)
    assert log_prob_correction.shape == (N_mc,)
    assert action_id.shape == (N_mc,)


def test_composite_bond_flip_selects_both_actions(composite_setup):
    state, action, mc_keys = composite_setup
    _, _, _, action_id = action(mc_keys, state)
    assert set(np.unique(np.asarray(action_id)).tolist()) == {0, 1}


def test_composite_bond_flip_dtypes_preserved(composite_setup):
    state, action, mc_keys = composite_setup
    new_state, allowed_move, _, _ = action(mc_keys, state)
    assert new_state.spins.dtype == state.spins.dtype
    assert allowed_move.dtype == jnp.bool_


def test_composite_bond_flip_move_matches_selected_action(composite_setup):
    """BondFlip always flips exactly 2 sites; SpinFlip always flips exactly 1."""
    state, action, mc_keys = composite_setup
    new_state, _, _, action_id = action(mc_keys, state)

    n_changed = np.asarray(state.spins != new_state.spins).sum(axis=-1)
    action_id = np.asarray(action_id)
    assert np.all(n_changed[action_id == 0] == 2)
    assert np.all(n_changed[action_id == 1] == 1)


def test_composite_bond_flip_sample_runs(composite_setup):
    state, action, mc_keys = composite_setup

    model = SpinRBM(hidden_units=1, dtype=jnp.float64, complex=True)
    dummy = SpinState(spins=jnp.ones((1, Ns), dtype=jnp.float64), lattice=state.lattice)
    params = model.init(jax.random.key(0), dummy)
    wf = WaveFunction(params=params, apply_fn=model.apply)

    out_state, log_amps, acceptance = sample(5, state, action, mc_keys, wf)

    assert log_amps.shape == (N_mc,)
    assert jnp.all(acceptance >= 0.0) and jnp.all(acceptance <= 1.0)
    assert out_state.spins.shape == state.spins.shape
