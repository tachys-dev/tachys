import jax
import jax.numpy as jnp

from tachys.lattice.ansatz.rbm import SpinRBM
from tachys.lattice.operator.local_estimator import compute_expectation, local_estimator
from tachys.lattice.spins.hamiltonians.heisenberg import heisenberg_square_pbc
from tachys.lattice.spins.spin_action import SpinFlip
from tachys.lattice.spins.spin_state import SpinState, init_config_fixed_magn
from tachys.montecarlo import sample
from tachys.optimizer import SR
from tachys.wavefunction import WaveFunction

L = 4
N = L * L
N_mc = 16


def test_sr_one_step_updates():
    """Regression test: SR optimizer updates after one step match main.py output."""
    H = heisenberg_square_pbc(L, J=1.0)

    model = SpinRBM(num_hidden=1, dtype=jnp.float64, complex=True)
    spins = init_config_fixed_magn(jax.random.key(1), N, sz=0, N_mc=N_mc)
    state = SpinState(spins=spins, Ns=N)
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
    """Regression test: 5-step SR loop energies and final params match main.py output."""
    H = heisenberg_square_pbc(L, J=1.0)

    model = SpinRBM(num_hidden=1, dtype=jnp.float64, complex=True)
    spins = init_config_fixed_magn(jax.random.key(1), N, sz=0, N_mc=N_mc)
    state = SpinState(spins=spins, Ns=N)
    params = model.init(jax.random.key(0), state)
    wf = WaveFunction(params=params, apply_fn=model.apply)

    action = SpinFlip()
    eta = 0.01
    optimizer = SR(diag_shift=1e-4, mode="complex")
    opt_state = optimizer.init(wf.params)

    expected_energies = [
        0.49845034798956,
        0.45578996859319,
        0.48775500310278,
        0.42747543083096,
        0.47270234592904,
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
    assert jnp.allclose(p["imag_linear"]["bias"],   jnp.array([0.00026084]),   atol=1e-7)
    assert jnp.allclose(p["imag_linear"]["kernel"], jnp.array([[-0.31364685]]), atol=1e-7)
    assert jnp.allclose(p["linear"]["bias"],        jnp.array([0.00138222]),   atol=1e-7)
    assert jnp.allclose(p["linear"]["kernel"], jnp.array([
        [-0.0969001 ],
        [ 0.02296981],
        [-0.23968046],
        [ 0.24635371],
        [ 0.2938857  ],
        [-0.28868271],
        [ 0.02190884],
        [ 0.3615717  ],
        [ 0.32401256],
        [-0.38193654],
        [ 0.44747092],
        [-0.05253436],
        [-0.46747823],
        [ 0.57370795],
        [ 0.31069702],
        [ 0.3254355  ],
    ]), atol=1e-7)


def test_sr_real_loop_5_steps():
    """Regression test: 5-step SR (real mode) loop energies and final params match expected output."""
    H = heisenberg_square_pbc(L, J=1.0)

    model = SpinRBM(num_hidden=1, dtype=jnp.float64, complex=False)
    spins = init_config_fixed_magn(jax.random.key(1), N, sz=0, N_mc=N_mc)
    state = SpinState(spins=spins, Ns=N)
    params = model.init(jax.random.key(0), state)
    wf = WaveFunction(params=params, apply_fn=model.apply)

    action = SpinFlip()
    eta = 0.01
    optimizer = SR(diag_shift=1e-4, mode="real")
    opt_state = optimizer.init(wf.params)

    expected_energies = [
        0.51014276542612,
        0.46348845408898,
        0.51022108460392,
        0.43410867837301,
        0.50005404843662,
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
    assert jnp.allclose(p["linear"]["bias"], jnp.array([-0.03069617]), atol=1e-7)
    assert jnp.allclose(p["linear"]["kernel"], jnp.array([
        [-0.20044304],
        [ 0.06880774],
        [-0.23611262],
        [ 0.2718081 ],
        [ 0.25545573],
        [-0.2915846 ],
        [ 0.00691176],
        [ 0.32821012],
        [ 0.2733232 ],
        [-0.37961912],
        [ 0.49910914],
        [-0.0374123 ],
        [-0.4160824 ],
        [ 0.52012254],
        [ 0.44771584],
        [ 0.26099295],
    ]), atol=1e-7)
