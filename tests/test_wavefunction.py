"""WaveFunction always hands the ansatz a batch of configurations.

The optimizers evaluate the ansatz on one configuration at a time (per-sample
Jacobians), so ``WaveFunction`` wraps ``apply_fn``: a single configuration is
evaluated as a batch of one and returns a single log-amplitude. The modules
below refuse unbatched input, so any path that bypasses the wrapper fails.
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from tachys.lattice.foundation.foundation_state import FermionFoundationState
from tachys.lattice.lattice_database import chain
from tachys.lattice.spins.spin_state import SpinState
from tachys.wavefunction import WaveFunction


class BatchOnly(nn.Module):
    """log psi(s) = w . s + i sum(s), for a batch of spin configurations only."""

    @nn.compact
    def __call__(self, state):
        s = state.spins
        assert s.ndim == 2, f"expected a batch of configurations, got shape {s.shape}"
        w = self.param("w", nn.initializers.normal(1.0), (s.shape[-1],), jnp.float64)
        return s.astype(jnp.float64) @ w + 1j * jnp.sum(s, axis=-1)


class FoundationBatchOnly(nn.Module):
    """Uses every data leaf of a foundation state, batched only."""

    @nn.compact
    def __call__(self, state):
        occ, couplings, ids = state.occupations, state.system_couplings, state.system_ids
        assert occ.ndim == 2 and couplings.ndim == 2 and ids.ndim == 1, (
            occ.shape, couplings.shape, ids.shape)
        w = self.param("w", nn.initializers.normal(1.0), (occ.shape[-1],), jnp.float64)
        v = self.param("v", nn.initializers.normal(1.0), (couplings.shape[-1],), jnp.float64)
        return occ @ w + couplings @ v + ids


def _spins():
    spins = jnp.array([[1, -1, 1, -1], [1, 1, -1, -1], [-1, 1, 1, -1]], dtype=jnp.int8)
    return SpinState(spins=spins, lattice=chain(4))


def _spin_wavefunction(batch):
    model = BatchOnly()
    return model, WaveFunction(params=model.init(jax.random.key(0), batch), apply_fn=model.apply)


def test_single_configuration_is_evaluated_as_a_batch_of_one():
    batch = _spins()
    _, wf = _spin_wavefunction(batch)
    batched = wf.apply_fn(wf.params, batch)
    assert batched.shape == (3,)
    for k in range(3):
        single = jax.tree.map(lambda x: x[k], batch)
        out = wf.apply_fn(wf.params, single)
        assert out.shape == ()
        np.testing.assert_allclose(out, batched[k])


def test_vmap_over_configurations_matches_the_batched_call():
    batch = _spins()
    _, wf = _spin_wavefunction(batch)
    np.testing.assert_allclose(jax.vmap(wf.apply_fn, in_axes=(None, 0))(wf.params, batch),
                               wf.apply_fn(wf.params, batch))


def test_every_leaf_of_a_foundation_state_gets_the_batch_axis():
    occ = jnp.array([[1, 0, 0, 1], [0, 1, 1, 0], [1, 1, 0, 0]], dtype=jnp.int32)
    batch = FermionFoundationState(
        occupations=occ, Ne=2, lattice=chain(2),
        system_couplings=jnp.array([[0.0], [4.0], [8.0]]),
        system_ids=jnp.array([0, 1, 2]), n_systems=3,
    )
    model = FoundationBatchOnly()
    wf = WaveFunction(params=model.init(jax.random.key(0), batch), apply_fn=model.apply)
    batched = wf.apply_fn(wf.params, batch)
    for k in range(3):
        single = jax.tree.map(lambda x: x[k], batch)   # system_ids is 0-d here
        np.testing.assert_allclose(wf.apply_fn(wf.params, single), batched[k])


def test_wrapping_is_idempotent_and_does_not_retrace():
    batch = _spins()
    model, wf = _spin_wavefunction(batch)
    grads = jax.tree.map(jnp.ones_like, wf.params)

    assert wf.apply_fn.__wrapped__ == model.apply
    for rebuilt in (wf.replace(params=grads), jax.tree.map(lambda x: x, wf),
                    wf.apply_gradients(grads, 0.1),
                    WaveFunction(params=wf.params, apply_fn=wf.apply_fn)):
        assert rebuilt.apply_fn is wf.apply_fn

    traces = []

    @jax.jit
    def log_psi(wf, state):
        traces.append(None)
        return wf.apply_fn(wf.params, state)

    log_psi(wf, batch)
    log_psi(wf.apply_gradients(grads, 0.1), batch)
    assert len(traces) == 1
