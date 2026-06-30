import jax
import jax.numpy as jnp

from tachys.lattice.ansatz.rbm import SpinRBM
from tachys.lattice.operator.local_estimator import compute_expectation
from tachys.lattice.spins.hamiltonians.heisenberg import heisenberg_square_pbc
from tachys.lattice.spins.spin_action import SpinFlip
from tachys.lattice.spins.spin_state import SpinState, init_config_fixed_magn
from tachys.montecarlo import sample
from tachys.optimizer import SPRING
from tachys.wavefunction import WaveFunction

L = 4
N = L * L
N_mc = 16


def test_spring_loop_5_steps():
    """Regression test: 5-step SPRING loop energies and final params match expected output."""
    H = heisenberg_square_pbc(L, J=1.0)

    model = SpinRBM(num_hidden=1, dtype=jnp.float64, complex=True)
    spins = init_config_fixed_magn(jax.random.key(1), N, sz=0, N_mc=N_mc)
    state = SpinState(spins=spins, Ns=N)
    params = model.init(jax.random.key(0), state)
    wf = WaveFunction(params=params, apply_fn=model.apply)

    action = SpinFlip()
    eta = 0.01
    optimizer = SPRING(diag_shift=1e-4, mode="complex", mu=0.9)
    opt_state = optimizer.init(wf.params)

    expected_energies = [
        0.49845034798956,
        0.45578996859319,
        0.48625662750767,
        0.42548762804911,
        0.50268573388225,
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
    assert jnp.allclose(p["imag_linear"]["bias"],   jnp.array([0.00158846]),   atol=1e-7)
    assert jnp.allclose(p["imag_linear"]["kernel"], jnp.array([[-0.3091556]]),  atol=1e-7)
    assert jnp.allclose(p["linear"]["bias"],        jnp.array([-0.00187971]),  atol=1e-7)
    assert jnp.allclose(p["linear"]["kernel"], jnp.array([
        [-0.01378087],
        [ 0.08473147],
        [-0.29646568],
        [ 0.21469285],
        [ 0.22388988],
        [-0.29666483],
        [-0.05074487],
        [ 0.32267591],
        [ 0.34443825],
        [-0.43681055],
        [ 0.42299276],
        [-0.0502227 ],
        [-0.4924946 ],
        [ 0.49282716],
        [ 0.38666276],
        [ 0.30037682],
    ]), atol=1e-7)


def test_spring_real_loop_5_steps():
    """Regression test: 5-step SPRING (real mode) loop energies and final params match expected output."""
    H = heisenberg_square_pbc(L, J=1.0)

    model = SpinRBM(num_hidden=1, dtype=jnp.float64, complex=False)
    spins = init_config_fixed_magn(jax.random.key(1), N, sz=0, N_mc=N_mc)
    state = SpinState(spins=spins, Ns=N)
    params = model.init(jax.random.key(0), state)
    wf = WaveFunction(params=params, apply_fn=model.apply)

    action = SpinFlip()
    eta = 0.01
    optimizer = SPRING(diag_shift=1e-4, mode="real", mu=0.9)
    opt_state = optimizer.init(wf.params)

    expected_energies = [
        0.51014276542612,
        0.46348845408898,
        0.49979477415191,
        0.44894247297603,
        0.50330481603641,
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
    assert jnp.allclose(p["linear"]["bias"], jnp.array([-0.04768529]), atol=1e-7)
    assert jnp.allclose(p["linear"]["kernel"], jnp.array([
        [-0.20778383],
        [-0.01692071],
        [-0.25202451],
        [ 0.26055509],
        [ 0.32571181],
        [-0.30236951],
        [ 0.00659781],
        [ 0.37524398],
        [ 0.29397111],
        [-0.390382  ],
        [ 0.51047904],
        [-0.05280636],
        [-0.40244537],
        [ 0.55792174],
        [ 0.39663766],
        [ 0.27021184],
    ]), atol=1e-7)
