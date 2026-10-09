import flax.linen as nn
import jax
import jax.numpy as jnp
import pytest

from testing_configs import frozen_config
from tachys.lattice.lattice_database import square
from tachys.lattice.operator.local_estimator import compute_expectation
from tachys.lattice.spins.hamiltonians.ising_transverse_field import ising_transverse_field_square_pbc
from tachys.lattice.spins.spin_action import SpinFlip
from tachys.lattice.spins.spin_state import SpinState
from tachys.montecarlo import sample
from tachys.optimizer import MARCH, SPRING, SR
from tachys.wavefunction import WaveFunction

L = 4
N_mc = 16


class ScalarJastrow(nn.Module):
    """log psi = a * sum_i s_i s_{i+1} + w . s, with a parameter `a` of shape ()."""

    @nn.compact
    def __call__(self, state):
        s = state.spins.astype(jnp.float64)
        a = self.param("a", nn.initializers.constant(0.1), (), jnp.float64)
        w = self.param("w", nn.initializers.constant(0.01), (s.shape[-1],), jnp.float64)
        return a * jnp.sum(s * jnp.roll(s, 1, axis=-1), axis=-1) + s @ w


@pytest.mark.parametrize("optimizer", [
    SR(diag_shift=1e-3, mode="real"),
    SPRING(diag_shift=1e-3, mode="real"),
    MARCH(diag_shift=1e-3, mode="real"),
], ids=["SR", "SPRING", "MARCH"])
def test_scalar_param_keeps_its_shape(optimizer):
    """Regression test: SPRING and MARCH returned (1,) updates for a shape-() parameter,
    broadcast against their (1,) mu, beta and step counter, so the next apply failed."""
    H = ising_transverse_field_square_pbc(L, J=1.0, h=1.0)
    state = SpinState(spins=frozen_config("square16_nmc16"), lattice=square(shape=(L, L)))
    model = ScalarJastrow()
    wf = WaveFunction(params=model.init(jax.random.key(0), state), apply_fn=model.apply)
    shapes = jax.tree.map(jnp.shape, wf.params)
    opt_state = optimizer.init(wf.params)

    for step in range(3):
        mc_keys = jax.random.split(jax.random.key(step), N_mc)
        state, log_amps, _acceptance = sample(1, state, SpinFlip(), mc_keys, wf)
        E_L, _e_mean, _ = compute_expectation(H, wf, state, log_amps)
        updates, opt_state = optimizer(E_L, opt_state, state, wf)
        assert jax.tree.map(jnp.shape, updates) == shapes, f"step {step}"
        wf = wf.apply_gradients(updates, 0.01)

    assert jax.tree.map(jnp.shape, wf.params) == shapes
