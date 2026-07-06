"""Tests that eloc/NTK/SR-solution centering use per-system means for
FoundationState, and fall back to the plain global-mean centering for a
regular State (tachys.optimizer.optimizers._center_eloc,
tachys.optimizer._kernels.center_ntk / center_sr_solution).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import shard_map
from jax.sharding import PartitionSpec as P

from tachys.lattice.ansatz.rbm_foundation import FermionFoundationRBM
from tachys.lattice.fermions.fermion_state import init_config_spinful
from tachys.lattice.fermions.hamiltonians.hubbard import hubbard_square_pbc
from tachys.lattice.foundation.foundation_state import FermionFoundationState
from tachys.lattice.foundation.operators import combine_systems, extract_system_couplings
from tachys.lattice.lattice_database import square
from tachys.lattice.operator.local_estimator import compute_expectation
from tachys.lattice.spins.spin_state import SpinState, init_config_fixed_magn
from tachys.optimizer._kernels import center_ntk, center_sr_solution, ntk_parallel_fn
from tachys.optimizer.optimizers import _center_eloc
from tachys.parallel import mesh
from tachys.wavefunction import WaveFunction

L = 4
N = L * L
Ne = 10
Us = [0, 2, 4, 8]
n_systems = len(Us)
N_mc_per_system = 8
N_mc = n_systems * N_mc_per_system


def _make_foundation_state_and_wf():
    lattice = square(shape=(L, L))
    Hs = [hubbard_square_pbc(L, U=U) for U in Us]
    H = combine_systems(Hs, N_mc_per_system)
    system_couplings = extract_system_couplings(H)
    system_ids = jnp.repeat(jnp.arange(n_systems), N_mc_per_system)

    initial_occupations, *_ = init_config_spinful(key=jax.random.key(0), Ne=Ne, Ns=N, N_mc=N_mc)
    state = FermionFoundationState(
        occupations=initial_occupations,
        lattice=lattice,
        Ne=Ne,
        system_couplings=system_couplings,
        system_ids=system_ids,
        n_systems=n_systems,
    )

    model = FermionFoundationRBM(hidden_units=8)
    params = model.init(jax.random.key(1), state)
    wf = WaveFunction(params=params, apply_fn=model.apply, dtype=jnp.float64)
    return H, state, wf


def _reference_double_center(ntk, sys_ids, K):
    """Per-system row/col/global-mean centering, computed independently of collectives.grouped_mean."""
    n = ntk.shape[0]
    row_mean = np.stack([ntk[sys_ids == sys_ids[i]].mean(axis=0) for i in range(n)])
    col_mean = np.stack([ntk[:, sys_ids == sys_ids[j]].mean(axis=1) for j in range(n)], axis=1)
    global_mean = np.array([
        [ntk[np.ix_(sys_ids == sys_ids[i], sys_ids == sys_ids[j])].mean() for j in range(n)]
        for i in range(n)
    ])
    return ntk - row_mean - col_mean + global_mean


# ─── _center_eloc ───────────────────────────────────────────────────────────

def test_center_eloc_foundation_state_uses_per_system_mean():
    H, state, wf = _make_foundation_state_and_wf()
    log_amps = wf.apply_fn(wf.params, state)
    E_L, _, _ = compute_expectation(H, wf, state, log_amps)

    eloc = shard_map(
        _center_eloc, mesh=mesh, in_specs=(P('i'), P('i')), out_specs=P('i'), check_vma=False,
    )(E_L, state)

    system_ids = state.system_ids
    for g in range(n_systems):
        assert jnp.allclose(jnp.mean(eloc[system_ids == g]), 0.0, atol=1e-10)

    expected = E_L - jnp.stack(
        [jnp.mean(E_L[system_ids == g]) for g in range(n_systems)]
    )[system_ids]
    assert jnp.allclose(eloc, expected, atol=1e-10)


def test_center_eloc_plain_state_uses_global_mean():
    lattice = square(shape=(L, L))
    spins = init_config_fixed_magn(jax.random.key(1), N, sz=0, N_mc=N_mc)
    state = SpinState(spins=spins, lattice=lattice)
    E_L = jax.random.normal(jax.random.key(2), (N_mc,))

    eloc = shard_map(
        _center_eloc, mesh=mesh, in_specs=(P('i'), P('i')), out_specs=P('i'), check_vma=False,
    )(E_L, state)

    assert jnp.allclose(eloc, E_L - jnp.mean(E_L), atol=1e-10)


# ─── center_ntk ─────────────────────────────────────────────────────────────

def test_center_ntk_foundation_state_matches_reference():
    _, state, wf = _make_foundation_state_and_wf()

    def _raw_and_centered(state, wf):
        raw = ntk_parallel_fn(state, wf, 1, "real")
        return raw, center_ntk(raw, None, state)

    raw_ntk, centered_ntk = shard_map(
        _raw_and_centered, mesh=mesh,
        in_specs=(P('i'), P()), out_specs=(P(), P()), check_vma=False,
    )(state, wf)

    system_ids = np.asarray(state.system_ids)
    expected = _reference_double_center(np.asarray(raw_ntk), system_ids, n_systems)
    assert jnp.allclose(centered_ntk, expected, atol=1e-7)

    # Each system's own diagonal block should have ~zero row/col sums after
    # centering -- the per-system analogue of the global-mean invariant.
    centered_np = np.asarray(centered_ntk)
    for g in range(n_systems):
        mask = system_ids == g
        block = centered_np[np.ix_(mask, mask)]
        assert np.max(np.abs(block.sum(axis=0))) < 1e-6
        assert np.max(np.abs(block.sum(axis=1))) < 1e-6


def test_center_ntk_weights_with_foundation_state_raises():
    _, state, wf = _make_foundation_state_and_wf()
    weights = jnp.ones(state.config.shape[0])

    def _call(state, wf, weights):
        raw = ntk_parallel_fn(state, wf, 1, "real")
        return center_ntk(raw, weights, state)

    with pytest.raises(NotImplementedError):
        shard_map(
            _call, mesh=mesh,
            in_specs=(P('i'), P(), P('i')), out_specs=P(), check_vma=False,
        )(state, wf, weights)


# ─── center_sr_solution ─────────────────────────────────────────────────────

def test_center_sr_solution_foundation_state_matches_reference():
    _, state, wf = _make_foundation_state_and_wf()
    sr_solution_raw = jax.random.normal(jax.random.key(3), (N_mc,))

    centered = shard_map(
        lambda sol, s: center_sr_solution(sol, s, "real", None),
        mesh=mesh, in_specs=(P(), P('i')), out_specs=P(), check_vma=False,
    )(sr_solution_raw, state)

    system_ids = state.system_ids
    mean_a = jnp.stack(
        [jnp.mean(sr_solution_raw[system_ids == g]) for g in range(n_systems)]
    )[system_ids]
    expected = (sr_solution_raw - mean_a) / N_mc ** 0.5
    assert jnp.allclose(centered, expected, atol=1e-10)


def test_center_sr_solution_weights_with_foundation_state_raises():
    _, state, wf = _make_foundation_state_and_wf()
    sr_solution_raw = jax.random.normal(jax.random.key(4), (N_mc,))
    weights = jnp.ones(state.config.shape[0])

    with pytest.raises(NotImplementedError):
        shard_map(
            lambda sol, s, w: center_sr_solution(sol, s, "real", w),
            mesh=mesh, in_specs=(P(), P('i'), P('i')), out_specs=P(), check_vma=False,
        )(sr_solution_raw, state, weights)
