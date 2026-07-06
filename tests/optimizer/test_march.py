import jax
import jax.numpy as jnp

from tachys.lattice.ansatz.rbm import SpinRBM
from tachys.lattice.operator.local_estimator import compute_expectation
from tachys.lattice.spins.hamiltonians.ising_transverse_field import ising_transverse_field_square_pbc
from tachys.lattice.spins.spin_action import SpinFlip
from tachys.lattice.spins.spin_state import SpinState, init_config_fixed_magn
from tachys.lattice.lattice_database import square
from tachys.montecarlo import sample
from tachys.optimizer import MARCH
from tachys.wavefunction import WaveFunction

L = 4
N = L * L
N_mc = 16


def test_march_loop_5_steps():
    """Regression test: 5-step MARCH loop energies and final params match expected output for Ising model."""
    H = ising_transverse_field_square_pbc(L, J=1.0, h=1.0)

    model = SpinRBM(hidden_units=1, dtype=jnp.float64, complex=True)
    lattice = square(shape=(L, L))
    spins = init_config_fixed_magn(jax.random.key(1), N, sz=0, N_mc=N_mc)
    state = SpinState(spins=spins, lattice=lattice)
    params = model.init(jax.random.key(0), state)
    wf = WaveFunction(params=params, apply_fn=model.apply)

    action = SpinFlip()
    eta = 0.01
    optimizer = MARCH(diag_shift=1e-4, mode="complex")
    opt_state = optimizer.init(wf.params)

    expected_energies = [
        -0.99646772696840,
        -0.95977627718655,
        -1.05912357339741,
        -1.09557703113578,
        -1.12124446696711,
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
    assert jnp.allclose(p["imag_linear"]["bias"],   jnp.array([-0.02887906]),   atol=1e-7)
    assert jnp.allclose(p["imag_linear"]["kernel"], jnp.array([[-0.22720453]]),  atol=1e-7)
    assert jnp.allclose(p["linear"]["bias"],        jnp.array([-0.36542121]),   atol=1e-7)
    assert jnp.allclose(p["linear"]["kernel"], jnp.array([
        [-0.25426956],
        [-0.32711724],
        [ 0.15680319],
        [ 0.315573  ],
        [ 0.5517267 ],
        [ 0.10010998],
        [-0.12029529],
        [ 0.49503137],
        [ 0.37076275],
        [-0.04861747],
        [ 0.60071582],
        [ 0.20683792],
        [-0.02475675],
        [ 0.24742742],
        [ 0.49294378],
        [ 0.26559121],
    ]), atol=1e-7)


def test_march_real_loop_5_steps():
    """Regression test: 5-step MARCH (real mode) loop energies and final params match expected output for Ising model."""
    H = ising_transverse_field_square_pbc(L, J=1.0, h=1.0)

    model = SpinRBM(hidden_units=1, dtype=jnp.float64, complex=False)
    lattice = square(shape=(L, L))
    spins = init_config_fixed_magn(jax.random.key(1), N, sz=0, N_mc=N_mc)
    state = SpinState(spins=spins, lattice=lattice)
    params = model.init(jax.random.key(0), state)
    wf = WaveFunction(params=params, apply_fn=model.apply)

    action = SpinFlip()
    eta = 0.01
    optimizer = MARCH(diag_shift=1e-4, mode="real")
    opt_state = optimizer.init(wf.params)

    expected_energies = [
        -1.00735553414256,
        -0.97755924109531,
        -1.18543775301394,
        -1.01607240151782,
        -1.20960255198990,
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
    assert jnp.allclose(p["linear"]["bias"], jnp.array([-0.00339095]), atol=1e-7)
    assert jnp.allclose(p["linear"]["kernel"], jnp.array([
        [-0.07195265],
        [ 0.0509864 ],
        [ 0.07273569],
        [-0.0789607 ],
        [ 0.1309594 ],
        [-0.2678557 ],
        [ 0.08308403],
        [ 0.7976987 ],
        [ 0.2848133 ],
        [-0.12778808],
        [ 0.74112685],
        [ 0.2107187 ],
        [-0.44071884],
        [ 0.09984582],
        [ 0.72999138],
        [ 0.16289594],
    ]), atol=1e-7)
