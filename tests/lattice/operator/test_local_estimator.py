import dataclasses

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import PartitionSpec as P

from tachys.lattice.ansatz.rbm import SpinRBM
from tachys.lattice.foundation.operators import combine_systems
from tachys.lattice.operator.base import _OperatorSum, _OperatorMul
from tachys.lattice.operator.local_estimator import local_estimator, _operator_in_spec
from tachys.lattice.spins.hamiltonians.heisenberg import heisenberg_square_pbc
from tachys.lattice.spins.spin_state import SpinState, init_config_fixed_magn
from tachys.lattice.lattice_database import square
from tachys.wavefunction import WaveFunction

L = 4
N = L * L
N_mc = 16


@pytest.fixture(scope="module")
def setup():
    key = jax.random.key(0)

    H = heisenberg_square_pbc(L, J=1.0)

    lattice = square(shape=(L, L))
    key, k1, k2 = jax.random.split(key, 3)
    spins = init_config_fixed_magn(k1, N, sz=0, N_mc=N_mc)
    state = SpinState(spins=spins, lattice=lattice)

    model = SpinRBM(hidden_units=N, dtype=jnp.float64)
    dummy = SpinState(spins=jnp.ones((1, N), dtype=jnp.float64), lattice=lattice)
    params = model.init(k2, dummy)
    wf = WaveFunction(params=params, apply_fn=model.apply)

    log_amps = wf.apply_fn(wf.params, state)

    return H, state, wf, log_amps


def test_optimize_mask_matches_no_mask(setup):
    H, state, wf, log_amps = setup
    ref = local_estimator(H, state, wf, log_amps, optimize_mask=False)
    out = local_estimator(H, state, wf, log_amps, optimize_mask=True, batch_expand=1)
    assert jnp.allclose(ref, out, atol=1e-10)


def test_batch_expand_2_matches_no_mask(setup):
    H, state, wf, log_amps = setup
    ref = local_estimator(H, state, wf, log_amps, optimize_mask=False)
    out = local_estimator(H, state, wf, log_amps, optimize_mask=True, batch_expand=2)
    assert jnp.allclose(ref, out, atol=1e-10)


def test_batch_expand_1_matches_batch_expand_2(setup):
    H, state, wf, log_amps = setup
    out1 = local_estimator(H, state, wf, log_amps, optimize_mask=True, batch_expand=1)
    out2 = local_estimator(H, state, wf, log_amps, optimize_mask=True, batch_expand=2)
    assert jnp.allclose(out1, out2, atol=1e-10)


def test_local_estimator_hidden_units_1_values():
    """Regression test: O_L values for hidden_units=1, keys matching main.py."""
    H = heisenberg_square_pbc(L, J=1.0)

    lattice = square(shape=(L, L))
    spins = init_config_fixed_magn(jax.random.key(1), N, sz=0, N_mc=N_mc)
    state = SpinState(spins=spins, lattice=lattice)

    model = SpinRBM(hidden_units=1, dtype=jnp.float64)
    dummy = SpinState(spins=jnp.ones((1, N), dtype=jnp.float64), lattice=lattice)
    params = model.init(jax.random.key(0), dummy)
    wf = WaveFunction(params=params, apply_fn=model.apply)

    log_amps = wf.apply_fn(wf.params, state)
    O_L = local_estimator(H, state, wf, log_amps, optimize_mask=False)

    expected = jnp.array([
        10.08800221+0.j,  6.67422546+0.j,  8.98978462+0.j,  8.16693411+0.j,
        10.96656328+0.j,  9.99960608+0.j,  8.8268125 +0.j, 10.01528845+0.j,
         9.92333354+0.j,  8.13610393+0.j, 10.92827389+0.j, 11.34194165+0.j,
         7.94093406+0.j,  6.39475996+0.j,  5.67186314+0.j, 11.30879845+0.j,
    ])
    assert jnp.allclose(O_L, expected, atol=1e-8)


def test_local_estimator_hidden_units_1_complex_values():
    """Regression test: O_L values for hidden_units=1, complex=True, keys matching main.py."""
    H = heisenberg_square_pbc(L, J=1.0)

    lattice = square(shape=(L, L))
    spins = init_config_fixed_magn(jax.random.key(1), N, sz=0, N_mc=N_mc)
    state = SpinState(spins=spins, lattice=lattice)

    model = SpinRBM(hidden_units=1, dtype=jnp.float64, complex=True)
    dummy = SpinState(spins=jnp.ones((1, N), dtype=jnp.float64), lattice=lattice)
    params = model.init(jax.random.key(0), dummy)
    wf = WaveFunction(params=params, apply_fn=model.apply)

    log_amps = wf.apply_fn(wf.params, state)
    O_L = local_estimator(H, state, wf, log_amps, optimize_mask=False)

    expected = jnp.array([
         9.87823677-1.24439457j,  6.49979171+0.10703017j,  8.82752373-0.66993061j,
         8.09436694-0.18338328j, 10.62071174-1.82094116j,  9.83360165-1.14484738j,
         8.6972422 -0.57128486j,  9.69838734-1.32275405j,  9.65330609-1.23787047j,
         8.03965236-0.20537671j, 10.56732227-1.82047406j, 10.8912091 -2.11758494j,
         7.82962884-0.14988727j,  6.26067337+0.3034345j,   5.54019536+0.52877809j,
        10.84961047-2.10761736j,
    ])
    assert jnp.allclose(O_L, expected, atol=1e-8)


def _iter_leaf_operators(operator):
    """Depth-first leaf operators of an _OperatorSum/_OperatorMul tree."""
    if isinstance(operator, (_OperatorSum, _OperatorMul)):
        for op in operator.operators:
            yield from _iter_leaf_operators(op)
    else:
        yield operator


def test_operator_in_spec_replicates_plain_operator():
    """A plain (non-foundation) Hamiltonian has 1D couplings everywhere, so
    every leaf -- coupling included -- must stay fully replicated (P())."""
    H = heisenberg_square_pbc(L, J=1.0)
    spec = _operator_in_spec(H)

    for leaf_op in _iter_leaf_operators(spec):
        for f in dataclasses.fields(leaf_op):
            if not f.metadata.get('pytree_node', True):
                continue
            assert getattr(leaf_op, f.name) == P()


def test_operator_in_spec_shards_coupling_for_foundation_operator():
    """A combine_systems-combined Hamiltonian has 2D (n_terms, N_mc) couplings
    -- those must be sharded (P(None, 'i')); every other field must stay P()."""
    Hs = [heisenberg_square_pbc(L, J=J) for J in (1.0, 2.0)]
    H_comb = combine_systems(Hs, 4)
    spec = _operator_in_spec(H_comb)

    for leaf_op in _iter_leaf_operators(spec):
        for f in dataclasses.fields(leaf_op):
            if not f.metadata.get('pytree_node', True):
                continue
            expected = P(None, 'i') if f.name == 'coupling' else P()
            assert getattr(leaf_op, f.name) == expected
