import jax
import jax.numpy as jnp
import pytest

from tachys.lattice.ansatz.rbm import SpinRBM
from tachys.lattice.operator.local_estimator import local_estimator
from tachys.lattice.spins.hamiltonians.heisenberg import heisenberg_square_pbc
from tachys.lattice.spins.spin_state import SpinState, init_config_fixed_magn
from tachys.wavefunction import WaveFunction

L = 4
N = L * L
N_mc = 16


@pytest.fixture(scope="module")
def setup():
    key = jax.random.key(0)

    H = heisenberg_square_pbc(L, J=1.0)

    key, k1, k2 = jax.random.split(key, 3)
    spins = init_config_fixed_magn(k1, N, sz=0, N_mc=N_mc)
    state = SpinState(spins=spins, Ns=N)

    model = SpinRBM(num_hidden=N, dtype=jnp.float64)
    dummy = SpinState(spins=jnp.ones((1, N), dtype=jnp.float64), Ns=N)
    params = model.init(k2, dummy)
    wf = WaveFunction(params=params, apply_fn=model.apply)

    log_amp = wf.apply_fn(wf.params, state)

    return H, state, wf, log_amp


def test_optimize_mask_matches_no_mask(setup):
    H, state, wf, log_amp = setup
    ref = local_estimator(H, state, wf, log_amp, optimize_mask=False)
    out = local_estimator(H, state, wf, log_amp, optimize_mask=True, batch_expand=1)
    assert jnp.allclose(ref, out, atol=1e-10)


def test_batch_expand_2_matches_no_mask(setup):
    H, state, wf, log_amp = setup
    ref = local_estimator(H, state, wf, log_amp, optimize_mask=False)
    out = local_estimator(H, state, wf, log_amp, optimize_mask=True, batch_expand=2)
    assert jnp.allclose(ref, out, atol=1e-10)


def test_batch_expand_1_matches_batch_expand_2(setup):
    H, state, wf, log_amp = setup
    out1 = local_estimator(H, state, wf, log_amp, optimize_mask=True, batch_expand=1)
    out2 = local_estimator(H, state, wf, log_amp, optimize_mask=True, batch_expand=2)
    assert jnp.allclose(out1, out2, atol=1e-10)
