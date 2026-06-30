import jax
import jax.numpy as jnp

from tachys.lattice.ansatz.rbm import SpinRBM
from tachys.lattice.operator.local_estimator import compute_expectation
from tachys.lattice.spins.hamiltonians.heisenberg import heisenberg_square_pbc
from tachys.lattice.spins.spin_action import SpinFlip
from tachys.lattice.spins.spin_state import SpinState, init_config_fixed_magn
from tachys.montecarlo import sample
from tachys.optimizer import MARCH
from tachys.wavefunction import WaveFunction

L = 4
N = L * L
N_mc = 16


def test_march_loop_5_steps():
    """Regression test: 5-step MARCH loop energies and final params match expected output."""
    H = heisenberg_square_pbc(L, J=1.0)

    model = SpinRBM(num_hidden=1, dtype=jnp.float64, complex=True)
    spins = init_config_fixed_magn(jax.random.key(1), N, sz=0, N_mc=N_mc)
    state = SpinState(spins=spins, Ns=N)
    params = model.init(jax.random.key(0), state)
    wf = WaveFunction(params=params, apply_fn=model.apply)

    action = SpinFlip()
    eta = 0.01
    optimizer = MARCH(diag_shift=1e-4, mode="complex")
    opt_state = optimizer.init(wf.params)

    expected_energies = [
        0.49845034798956,
        0.44590277958980,
        0.48389267249104,
        0.41393401277404,
        0.46692398115314,
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
    assert jnp.allclose(p["imag_linear"]["bias"],   jnp.array([0.00140348]),   atol=1e-7)
    assert jnp.allclose(p["imag_linear"]["kernel"], jnp.array([[-0.31583106]]), atol=1e-7)
    assert jnp.allclose(p["linear"]["bias"],        jnp.array([-0.04930508]),  atol=1e-7)
    assert jnp.allclose(p["linear"]["kernel"], jnp.array([
        [-0.11547775],
        [ 0.02872157],
        [-0.27925555],
        [ 0.24793144],
        [ 0.30436625],
        [-0.24803722],
        [-0.02656085],
        [ 0.34678612],
        [ 0.35336046],
        [-0.42090048],
        [ 0.44445657],
        [-0.01075493],
        [-0.47022865],
        [ 0.50064603],
        [ 0.37501826],
        [ 0.25123394],
    ]), atol=1e-7)


def test_march_real_loop_5_steps():
    """Regression test: 5-step MARCH (real mode) loop energies and final params match expected output."""
    H = heisenberg_square_pbc(L, J=1.0)

    model = SpinRBM(num_hidden=1, dtype=jnp.float64, complex=False)
    spins = init_config_fixed_magn(jax.random.key(1), N, sz=0, N_mc=N_mc)
    state = SpinState(spins=spins, Ns=N)
    params = model.init(jax.random.key(0), state)
    wf = WaveFunction(params=params, apply_fn=model.apply)

    action = SpinFlip()
    eta = 0.01
    optimizer = MARCH(diag_shift=1e-4, mode="real")
    opt_state = optimizer.init(wf.params)

    expected_energies = [
        0.51014276542612,
        0.45415898960315,
        0.49058746290787,
        0.47440324845054,
        0.49346470223269,
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
    assert jnp.allclose(p["linear"]["bias"], jnp.array([0.01055326]), atol=1e-7)
    assert jnp.allclose(p["linear"]["kernel"], jnp.array([
        [-0.12773543],
        [ 0.096223  ],
        [-0.26214155],
        [ 0.25722229],
        [ 0.29212546],
        [-0.27545855],
        [ 0.02598533],
        [ 0.38496348],
        [ 0.3069271 ],
        [-0.40242077],
        [ 0.45538541],
        [-0.03831465],
        [-0.4762213 ],
        [ 0.557117  ],
        [ 0.37031073],
        [ 0.26536152],
    ]), atol=1e-7)
