import jax
import jax.numpy as jnp

from tachys.lattice.ansatz.rbm import SpinRBM
from tachys.lattice.operator.local_estimator import compute_expectation
from tachys.lattice.spins.hamiltonians.ising_transverse_field import ising_transverse_field_square_pbc
from tachys.lattice.spins.spin_action import SpinFlip
from tachys.lattice.spins.spin_state import SpinState, init_config_fixed_magn
from tachys.lattice.lattice_database import square
from tachys.montecarlo import sample
from tachys.optimizer import SPRING
from tachys.wavefunction import WaveFunction

L = 4
N = L * L
N_mc = 16


def test_spring_loop_5_steps():
    """Regression test: 5-step SPRING loop energies and final params match expected output for Ising model."""
    H = ising_transverse_field_square_pbc(L, J=1.0, h=1.0)

    model = SpinRBM(num_hidden=1, dtype=jnp.float64, complex=True)
    lattice = square(shape=(L, L))
    spins = init_config_fixed_magn(jax.random.key(1), N, sz=0, N_mc=N_mc)
    state = SpinState(spins=spins, lattice=lattice)
    params = model.init(jax.random.key(0), state)
    wf = WaveFunction(params=params, apply_fn=model.apply)

    action = SpinFlip()
    eta = 0.01
    optimizer = SPRING(diag_shift=1e-4, mode="complex", mu=0.9)
    opt_state = optimizer.init(wf.params)

    expected_energies = [
        -0.99646772696840,
        -0.92262547725766,
        -1.09651551660695,
        -1.02391125310247,
        -1.05780590445704,
    ]

    for step in range(5):
        mc_keys = jax.random.split(jax.random.key(2), N_mc)
        state, log_amps, _acceptance = sample(1, state, action, mc_keys, wf)
        E_L, e_mean, _ = compute_expectation(H, wf, state, log_amps)
        updates, opt_state = optimizer(E_L, opt_state, state, wf)
        wf = wf.apply_gradients(updates, eta)

        assert jnp.isclose(e_mean.real / N, expected_energies[step], atol=1e-12), \
            f"step {step}: expected {expected_energies[step]}, got {float(e_mean.real / N):.14f}"

    p = wf.params["params"]
    assert jnp.allclose(p["imag_linear"]["bias"],   jnp.array([-0.02578163]),   atol=1e-7)
    assert jnp.allclose(p["imag_linear"]["kernel"], jnp.array([[-0.1676257]]),   atol=1e-7)
    assert jnp.allclose(p["linear"]["bias"],        jnp.array([-0.15544766]),   atol=1e-7)
    assert jnp.allclose(p["linear"]["kernel"], jnp.array([
        [ 0.18228742],
        [-0.24142391],
        [-0.05896187],
        [ 0.2256695 ],
        [ 0.35952003],
        [-0.15245578],
        [ 0.0145417 ],
        [ 0.5528769 ],
        [ 0.12113736],
        [-0.24276977],
        [ 0.60893073],
        [ 0.29019877],
        [-0.16040583],
        [ 0.23057249],
        [ 0.32995867],
        [ 0.45889089],
    ]), atol=1e-7)


def test_spring_real_loop_5_steps():
    """Regression test: 5-step SPRING (real mode) loop energies and final params match expected output for Ising model."""
    H = ising_transverse_field_square_pbc(L, J=1.0, h=1.0)

    model = SpinRBM(num_hidden=1, dtype=jnp.float64, complex=False)
    lattice = square(shape=(L, L))
    spins = init_config_fixed_magn(jax.random.key(1), N, sz=0, N_mc=N_mc)
    state = SpinState(spins=spins, lattice=lattice)
    params = model.init(jax.random.key(0), state)
    wf = WaveFunction(params=params, apply_fn=model.apply)

    action = SpinFlip()
    eta = 0.01
    optimizer = SPRING(diag_shift=1e-4, mode="real", mu=0.9)
    opt_state = optimizer.init(wf.params)

    expected_energies = [
        -1.00735553414256,
        -0.93472240035012,
        -1.12694723517243,
        -0.87005214379056,
        -1.13877833321891,
    ]

    for step in range(5):
        mc_keys = jax.random.split(jax.random.key(2), N_mc)
        state, log_amps, _acceptance = sample(1, state, action, mc_keys, wf)
        E_L, e_mean, _ = compute_expectation(H, wf, state, log_amps)
        updates, opt_state = optimizer(E_L, opt_state, state, wf)
        wf = wf.apply_gradients(updates, eta)

        assert jnp.isclose(e_mean.real / N, expected_energies[step], atol=1e-12), \
            f"step {step}: expected {expected_energies[step]}, got {float(e_mean.real / N):.14f}"

    p = wf.params["params"]
    assert jnp.allclose(p["linear"]["bias"], jnp.array([-0.41525903]), atol=1e-7)
    assert jnp.allclose(p["linear"]["kernel"], jnp.array([
        [-0.02687334],
        [-0.40836172],
        [-0.39555034],
        [ 0.15339498],
        [ 0.35678344],
        [-0.1961157 ],
        [ 0.04453948],
        [ 0.39765335],
        [-0.04488276],
        [-0.40713003],
        [ 1.25552396],
        [ 0.17200909],
        [ 0.07376105],
        [ 0.35774067],
        [ 0.53686779],
        [ 0.57626538],
    ]), atol=1e-7)
