import jax
import jax.numpy as jnp

from tachys.lattice.ansatz.rbm import SpinRBM
from tachys.lattice.operator.local_estimator import compute_expectation, local_estimator
from tachys.lattice.spins.hamiltonians.heisenberg import heisenberg_square_pbc
from tachys.lattice.spins.hamiltonians.ising_transverse_field import ising_transverse_field_square_pbc
from tachys.lattice.spins.spin_action import SpinFlip
from tachys.lattice.spins.spin_state import SpinState, init_config_fixed_magn
from tachys.lattice.lattice_database import square
from tachys.montecarlo import sample
from tachys.optimizer import SR
from tachys.wavefunction import WaveFunction

L = 4
N = L * L
N_mc = 16


def test_sr_one_step_updates():
    """Regression test: SR optimizer updates after one step match main.py output."""
    H = heisenberg_square_pbc(L, J=1.0) #! ising should be used

    model = SpinRBM(num_hidden=1, dtype=jnp.float64, complex=True)
    lattice = square(shape=(L, L))
    spins = init_config_fixed_magn(jax.random.key(1), N, sz=0, N_mc=N_mc)
    state = SpinState(spins=spins, lattice=lattice)
    params = model.init(jax.random.key(0), state)
    wf = WaveFunction(params=params, apply_fn=model.apply)

    optimizer = SR(diag_shift=1e-4, mode="complex")
    opt_state = optimizer.init(wf.params)

    mc_keys = jax.random.split(jax.random.key(2), N_mc)
    state, log_amps, _acceptance = sample(1, state, SpinFlip(), mc_keys, wf)
    O_L = local_estimator(H, state, wf, log_amps)
    updates, _opt_state = optimizer(O_L, opt_state, state, wf)

    expected_imag_bias = jnp.array([0.05400766])
    expected_imag_kernel = jnp.array([[0.38214008]])
    expected_linear_bias = jnp.array([2.83138641])
    expected_linear_kernel = jnp.array([
        [ 3.13913137],
        [ 2.86079906],
        [-0.23305401],
        [-0.23735038],
        [-0.85217398],
        [ 1.77201661],
        [ 2.32462515],
        [ 0.72903853],
        [ 0.21916408],
        [ 1.51565109],
        [-3.88390322],
        [ 1.18607806],
        [-0.67323989],
        [-0.45294976],
        [-4.24523083],
        [-1.72212511],
    ])

    p = updates["params"]
    assert jnp.allclose(p["imag_linear"]["bias"],   expected_imag_bias,    atol=1e-7)
    assert jnp.allclose(p["imag_linear"]["kernel"],  expected_imag_kernel,  atol=1e-7)
    assert jnp.allclose(p["linear"]["bias"],         expected_linear_bias,  atol=1e-7)
    assert jnp.allclose(p["linear"]["kernel"],       expected_linear_kernel, atol=1e-7)


def test_sr_loop_5_steps():
    """Regression test: 5-step SR loop energies and final params match expected output for Ising model."""
    H = ising_transverse_field_square_pbc(L, J=1.0, h=1.0)

    model = SpinRBM(num_hidden=1, dtype=jnp.float64, complex=True)
    lattice = square(shape=(L, L))
    spins = init_config_fixed_magn(jax.random.key(1), N, sz=0, N_mc=N_mc)
    state = SpinState(spins=spins, lattice=lattice)
    params = model.init(jax.random.key(0), state)
    wf = WaveFunction(params=params, apply_fn=model.apply)

    action = SpinFlip()
    eta = 0.01
    optimizer = SR(diag_shift=1e-4, mode="complex")
    opt_state = optimizer.init(wf.params)

    expected_energies = [
        -0.99646772696840,
        -0.92262547725766,
        -1.07355544444143,
        -1.03770217010731,
        -0.98556413159589,
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
    assert jnp.allclose(p["imag_linear"]["bias"],   jnp.array([-0.02153195]),   atol=1e-7)
    assert jnp.allclose(p["imag_linear"]["kernel"], jnp.array([[-0.16989735]]),  atol=1e-7)
    assert jnp.allclose(p["linear"]["bias"],        jnp.array([-0.09049892]),   atol=1e-7)
    assert jnp.allclose(p["linear"]["kernel"], jnp.array([
        [-0.00549538],
        [-0.57975204],
        [-0.42330984],
        [ 0.30260539],
        [ 0.42412032],
        [-0.26708997],
        [-0.02215112],
        [ 0.64839192],
        [ 0.0194746 ],
        [-0.621071  ],
        [-0.17652029],
        [ 0.06760028],
        [-0.09426484],
        [ 0.56902585],
        [ 0.48198705],
        [ 0.41160606],
    ]), atol=1e-7)


def test_sr_real_loop_5_steps():
    """Regression test: 5-step SR (real mode) loop energies and final params match expected output."""
    H = ising_transverse_field_square_pbc(L, J=1.0, h=1.0)

    model = SpinRBM(num_hidden=1, dtype=jnp.float64, complex=False)
    lattice = square(shape=(L, L))
    spins = init_config_fixed_magn(jax.random.key(1), N, sz=0, N_mc=N_mc)
    state = SpinState(spins=spins, lattice=lattice)
    params = model.init(jax.random.key(0), state)
    wf = WaveFunction(params=params, apply_fn=model.apply)

    action = SpinFlip()
    eta = 0.01
    optimizer = SR(diag_shift=1e-4, mode="real")
    opt_state = optimizer.init(wf.params)

    expected_energies = [
        -1.00735553414256,
        -0.93472240035012,
        -1.12213451032061,
        -0.96788517098408,
        -1.10386617541288,
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
    assert jnp.allclose(p["linear"]["bias"], jnp.array([-0.45330682]), atol=1e-7)
    assert jnp.allclose(p["linear"]["kernel"], jnp.array([
        [-0.37486956],
        [-0.4709006 ],
        [-0.07152381],
        [ 0.09834236],
        [ 0.34400194],
        [-0.25285732],
        [-0.05991222],
        [ 0.27908684],
        [ 0.06448295],
        [-0.17423916],
        [ 0.71773061],
        [ 0.16435237],
        [-0.06640457],
        [ 0.2647537 ],
        [ 0.43440253],
        [ 0.3993689 ],
    ]), atol=1e-7)
