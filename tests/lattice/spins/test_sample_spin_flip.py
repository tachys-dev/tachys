import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tachys.lattice.ansatz.rbm import SpinRBM
from tachys.lattice.spins.spin_action import SpinFlip
from tachys.lattice.spins.spin_state import SpinState, init_config_fixed_magn
from tachys.lattice.lattice_database import square
from tachys.montecarlo import sample
from tachys.utils import same_treedef_and_avals
from tachys.wavefunction import WaveFunction

L = 4
N = L * L
N_mc = 16


@pytest.fixture(scope="module")
def setup():
    lattice = square(shape=(L, L))
    spins = init_config_fixed_magn(jax.random.key(1), N, sz=0, N_mc=N_mc)
    state = SpinState(spins=spins, lattice=lattice)

    model = SpinRBM(num_hidden=1, dtype=jnp.float64, complex=True)
    dummy = SpinState(spins=jnp.ones((1, N), dtype=jnp.float64), lattice=lattice)
    params = model.init(jax.random.key(0), dummy)
    wf = WaveFunction(params=params, apply_fn=model.apply)

    action = SpinFlip()
    mc_keys = jax.random.split(jax.random.key(2), N_mc)
    return state, wf, action, mc_keys


def test_sample_returns_correct_shapes(setup):
    state, wf, action, mc_keys = setup
    out_state, log_amps, acceptance = sample(1, state, action, mc_keys, wf)
    assert log_amps.shape == (N_mc,)
    assert out_state.spins.shape == (N_mc, N)
    assert acceptance.shape == (1,)


def test_sample_acceptance_in_range(setup):
    state, wf, action, mc_keys = setup
    _, _, acceptance = sample(1, state, action, mc_keys, wf)
    assert jnp.all(acceptance >= 0.0)
    assert jnp.all(acceptance <= 1.0)


def test_sample_acceptance_regression(setup):
    state, wf, action, mc_keys = setup
    _, _, acceptance = sample(1, state, action, mc_keys, wf)
    assert jnp.allclose(acceptance, jnp.array([0.74609375]))


def test_sample_spins_valid_values(setup):
    state, wf, action, mc_keys = setup
    out_state, _, _ = sample(1, state, action, mc_keys, wf)
    unique = jnp.unique(out_state.spins)
    assert set(np.asarray(unique)).issubset({-1, 1})


def test_sample_1_sweep_regression():
    """Regression test: log-amplitudes and spins after 1 sweep with SpinFlip match main.py reference."""
    lattice = square(shape=(L, L))
    spins = init_config_fixed_magn(jax.random.key(1), N, sz=0, N_mc=N_mc)
    state = SpinState(spins=spins, lattice=lattice)

    model = SpinRBM(num_hidden=1, dtype=jnp.float64, complex=True)
    dummy = SpinState(spins=jnp.ones((1, N), dtype=jnp.float64), lattice=lattice)
    params = model.init(jax.random.key(0), dummy)
    wf = WaveFunction(params=params, apply_fn=model.apply)

    action = SpinFlip()
    mc_keys = jax.random.split(jax.random.key(2), N_mc)
    out_state, log_amps, _ = sample(1, state, action, mc_keys, wf)

    expected_log_amps = jnp.array([
        2.27277566e+00 - 8.53659259e-01j,
        1.66059344e+00 - 6.70053721e-01j,
        2.38684611e-01 - 1.35901276e-01j,
        9.95235327e-02 - 5.99579318e-02j,
        7.29698390e-01 - 3.54512861e-01j,
        3.28004681e-01 - 1.80659238e-01j,
        4.50974347e-01 - 2.38104627e-01j,
        1.50575393e-01 - 8.87906146e-02j,
        8.35181170e-01 - 3.94756531e-01j,
        9.39858569e-01 - 4.33104950e-01j,
        3.36565572e-01 - 1.84806565e-01j,
        2.97033211e-04 - 1.87053927e-04j,
        1.11918415e+00 - 4.95805042e-01j,
        6.77764725e-01 - 3.34022113e-01j,
        6.62059444e-01 - 3.27729206e-01j,
        7.40161704e-01 - 3.58584437e-01j,
    ])
    assert jnp.allclose(log_amps, expected_log_amps, atol=1e-7)

    expected_spins = jnp.array([
        [-1, -1, -1, -1, -1,  1,  1, -1, -1,  1, -1, -1,  1, -1, -1, -1],
        [-1,  1, -1,  1,  1,  1, -1,  1,  1, -1,  1, -1,  1,  1,  1,  1],
        [ 1, -1, -1, -1, -1,  1, -1, -1,  1, -1,  1, -1, -1,  1,  1, -1],
        [-1, -1, -1,  1, -1, -1,  1, -1,  1, -1, -1,  1,  1,  1,  1, -1],
        [-1,  1, -1, -1,  1, -1,  1, -1,  1, -1,  1,  1,  1,  1,  1, -1],
        [-1,  1, -1, -1, -1, -1, -1, -1,  1, -1, -1,  1,  1, -1, -1,  1],
        [-1, -1,  1,  1,  1,  1, -1,  1,  1, -1, -1,  1, -1, -1,  1,  1],
        [ 1, -1,  1, -1,  1,  1,  1, -1,  1, -1, -1, -1,  1,  1,  1, -1],
        [ 1,  1,  1, -1,  1, -1, -1, -1, -1, -1, -1,  1,  1, -1, -1,  1],
        [-1, -1, -1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1, -1,  1,  1],
        [-1, -1, -1, -1, -1,  1,  1,  1, -1,  1, -1, -1, -1, -1,  1, -1],
        [-1,  1, -1, -1,  1, -1,  1,  1, -1,  1, -1,  1, -1, -1, -1,  1],
        [-1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1, -1, -1, -1],
        [ 1, -1,  1,  1,  1, -1,  1, -1, -1, -1,  1,  1, -1,  1,  1, -1],
        [-1, -1,  1,  1, -1,  1,  1,  1, -1, -1,  1, -1, -1,  1,  1, -1],
        [ 1, -1,  1, -1, -1,  1, -1, -1, -1,  1, -1,  1, -1,  1,  1, -1],
    ], dtype=jnp.int8)
    assert jnp.array_equal(out_state.spins, expected_spins)


def test_sample_10_sweeps_regression():
    """Regression test: log-amplitudes and spins after 10 sweeps with SpinFlip."""
    lattice = square(shape=(L, L))
    spins = init_config_fixed_magn(jax.random.key(1), N, sz=0, N_mc=N_mc)
    state = SpinState(spins=spins, lattice=lattice)

    model = SpinRBM(num_hidden=1, dtype=jnp.float64, complex=True)
    dummy = SpinState(spins=jnp.ones((1, N), dtype=jnp.float64), lattice=lattice)
    params = model.init(jax.random.key(0), dummy)
    wf = WaveFunction(params=params, apply_fn=model.apply)

    action = SpinFlip()
    mc_keys = jax.random.split(jax.random.key(2), N_mc)
    out_state, log_amps, _ = sample(10, state, action, mc_keys, wf)

    expected_log_amps = jnp.array([
        0.61479608 - 0.30850473j,
        2.19445604 - 0.83056611j,
        2.03660045 - 0.78374914j,
        0.31058957 - 0.17214864j,
        2.60537692 - 0.95103089j,
        2.03954402 - 0.78462599j,
        0.61414626 - 0.30823732j,
        0.00525551 - 0.003302j,
        1.40913877 - 0.59126542j,
        1.49891746 - 0.61973524j,
        2.74779254 - 0.99248324j,
        0.42199184 - 0.22495842j,
        1.53470185 - 0.63097026j,
        1.25996518 - 0.54291533j,
        0.07668507 - 0.04665746j,
        0.62057855 - 0.31088057j,
    ])
    assert jnp.allclose(log_amps, expected_log_amps, atol=1e-7)

    expected_spins = jnp.array([
        [ 1, -1,  1, -1, -1, -1, -1,  1,  1,  1, -1,  1,  1,  1, -1, -1],
        [-1,  1, -1,  1,  1, -1,  1,  1,  1, -1,  1, -1,  1,  1,  1,  1],
        [-1,  1,  1, -1,  1,  1,  1, -1, -1,  1, -1,  1,  1, -1, -1, -1],
        [-1, -1, -1,  1, -1, -1, -1, -1, -1, -1,  1,  1,  1, -1, -1, -1],
        [-1,  1, -1, -1,  1, -1,  1,  1,  1, -1,  1,  1, -1,  1,  1,  1],
        [-1,  1,  1, -1, -1, -1,  1, -1, -1,  1, -1,  1,  1, -1, -1, -1],
        [ 1,  1, -1,  1,  1,  1,  1,  1, -1,  1, -1,  1, -1,  1,  1,  1],
        [-1,  1, -1,  1,  1,  1,  1, -1,  1,  1, -1,  1, -1, -1,  1, -1],
        [-1,  1,  1, -1,  1,  1,  1,  1, -1,  1, -1, -1,  1, -1, -1, -1],
        [ 1, -1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1,  1, -1, -1, -1],
        [-1, -1, -1,  1,  1, -1, -1,  1,  1, -1,  1,  1, -1,  1,  1,  1],
        [ 1,  1,  1,  1,  1, -1,  1, -1,  1,  1, -1, -1, -1,  1,  1,  1],
        [ 1, -1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1],
        [ 1,  1,  1, -1,  1, -1, -1,  1, -1,  1, -1,  1,  1, -1, -1, -1],
        [ 1, -1, -1,  1,  1,  1,  1,  1, -1,  1, -1, -1, -1,  1, -1,  1],
        [-1, -1,  1,  1, -1,  1, -1, -1,  1, -1,  1,  1,  1, -1, -1, -1],
    ], dtype=jnp.int8)
    assert jnp.array_equal(out_state.spins, expected_spins)


def test_sample_preserves_state_structure(setup):
    """After sampling, the output SpinState has the same treedef and avals as the input."""
    state, wf, action, mc_keys = setup
    out_state, _, _ = sample(1, state, action, mc_keys, wf)
    assert same_treedef_and_avals(state, out_state)
