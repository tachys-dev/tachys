from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from testing_ansatz import FermionFoundationRBM, frozen_params
from testing_configs import frozen_config
from tachys.lattice.bond_exchange import BondExchange
from tachys.lattice.fermions.hamiltonians.hubbard import hubbard_square_pbc
from tachys.lattice.foundation.foundation_state import FermionFoundationState
from tachys.lattice.foundation.operators import combine_systems, extract_system_couplings
from tachys.lattice.lattice_database import square
from tachys.lattice.operator.local_estimator import compute_expectation
from tachys.montecarlo import sample
from tachys.optimizer import MARCH, SR
from tachys.wavefunction import WaveFunction

DATA_DIR = Path(__file__).parent / "data"

L = 4
seed = 0
N = L * L
Ne = 10

Us = [0, 2, 4, 8]
n_systems = len(Us)

N_mc_per_system = 8
N_mc = n_systems * N_mc_per_system

EXPECTED_E_L = jnp.array([
     4.30637291e+00+3.00966919e-16j, -5.54223911e+01-5.91257891e-16j,
     4.85555513e+00-2.06494852e-15j, -1.18072208e+01+2.47572339e-17j,
     2.59719566e+01-4.19467039e-15j,  9.13176764e+00+6.38327652e-16j,
    -8.50964513e+00+3.91872983e-16j,  1.43758370e+01+2.31073455e-15j,
     1.88499666e+01-5.74351084e-15j, -1.18773507e+01+1.07494259e-15j,
     6.39033741e+00+6.95874617e-16j,  6.11666109e+01-3.45362111e-15j,
    -1.47592819e+01+9.45913022e-16j,  3.09947643e+01+3.24237197e-15j,
    -1.69053605e+01+5.85657350e-16j, -3.10881042e+01-2.09985770e-16j,
     1.53423180e+01+3.51855806e-16j,  4.11331145e+00-2.70605087e-15j,
    -8.65457183e+02-3.08318038e-14j,  3.75826480e+00+3.35074075e-16j,
     7.30137613e+00-4.92463120e-16j,  1.09572358e-01-1.10118258e-15j,
     6.86601181e+00-3.67461790e-16j,  9.43287269e+00-1.27057570e-15j,
    -1.93724343e+01+7.56791751e-16j,  2.62173129e+01+2.72465509e-15j,
     2.86712460e+01-4.00056962e-15j,  3.76624623e+00-1.11920559e-15j,
     2.56819368e+00+3.92575671e-15j,  4.79687039e+00+1.65480433e-16j,
    -4.80068136e+00-6.64412850e-16j,  1.97979048e+01+2.05624296e-15j,
])

EXPECTED_E_MEAN = -22.850468216514756 - 1.1963888969625417e-15j


def test_foundation_local_energies_and_mean():
    lattice = square(shape=(L, L))

    Hs = [hubbard_square_pbc(L, U=U) for U in Us]
    H = combine_systems(Hs, N_mc_per_system)
    system_couplings = extract_system_couplings(H)
    system_ids = jnp.repeat(jnp.arange(n_systems), N_mc_per_system)

    initial_occupations = frozen_config("foundation_square16_ne10_nmc32")
    state = FermionFoundationState(
        occupations=initial_occupations,
        lattice=lattice,
        Ne=Ne,
        system_couplings=system_couplings,
        system_ids=system_ids,
        n_systems=n_systems,
    )

    model = FermionFoundationRBM(hidden_units=64)
    params = frozen_params("foundation_hidden64")
    wf = WaveFunction(params=params, apply_fn=model.apply, dtype=jnp.float64)

    log_amps = wf.apply_fn(wf.params, state)
    E_L, e_mean, e2_mean = compute_expectation(H, wf, state, log_amps)

    assert jnp.allclose(E_L, EXPECTED_E_L, atol=1e-8)
    assert jnp.allclose(e_mean, EXPECTED_E_MEAN, atol=1e-8)


EXPECTED_LOG_AMPS_AFTER_SAMPLING = jnp.array([
    0.75561839+0.j        , 0.52308245+3.14159265j, 0.64347171+3.14159265j,
    1.1742796 +0.j        , 0.71682578+3.14159265j, 2.64404998+0.j        ,
    0.0874762 +0.j        , 1.00009468+3.14159265j, 2.15810102+0.j        ,
    0.97200804+0.j        , 4.47171202+0.j        , 1.2812792 +0.j        ,
    2.73259777+0.j        , 1.41486905+0.j        , 1.03992037+3.14159265j,
    0.89536427+0.j        , 3.6043704 +3.14159265j, 2.03988969+3.14159265j,
    4.2480647 +3.14159265j, 2.6699062 +3.14159265j, 3.60133859+3.14159265j,
    4.66951129+0.j        , 4.47007413+0.j        , 5.32951642+3.14159265j,
    3.96816102+3.14159265j, 3.22090147+3.14159265j, 5.38918393+0.j        ,
    4.47149876+3.14159265j, 5.10953888+0.j        , 6.26135443+3.14159265j,
    5.07391914+3.14159265j, 5.41466458+0.j        ,
])


def test_foundation_log_amps_after_sampling():
    """Regression test: log-amplitudes returned by sample() after one sweep,
    keys matching the first training-loop iteration of
    examples/hubbard_foundation_model/main.py."""
    lattice = square(shape=(L, L))

    Hs = [hubbard_square_pbc(L, U=U) for U in Us]
    H = combine_systems(Hs, N_mc_per_system)
    system_couplings = extract_system_couplings(H)
    system_ids = jnp.repeat(jnp.arange(n_systems), N_mc_per_system)

    initial_occupations = frozen_config("foundation_square16_ne10_nmc32")
    state = FermionFoundationState(
        occupations=initial_occupations,
        lattice=lattice,
        Ne=Ne,
        system_couplings=system_couplings,
        system_ids=system_ids,
        n_systems=n_systems,
    )

    model = FermionFoundationRBM(hidden_units=64)
    params = frozen_params("foundation_hidden64")
    wf = WaveFunction(params=params, apply_fn=model.apply, dtype=jnp.float64)

    action = BondExchange.create(lattice, max_dist=1, Nbands=2)

    mc_keys = jax.random.split(jax.random.key(2), N_mc)
    state, log_amps, acceptance = sample(1, state, action, mc_keys, wf)

    assert jnp.allclose(log_amps, EXPECTED_LOG_AMPS_AFTER_SAMPLING, atol=1e-8)


_MARCH_FIXTURE = np.load(DATA_DIR / "march_first_step_updates.npz")
EXPECTED_MARCH_DENSE0_BIAS   = jnp.asarray(_MARCH_FIXTURE["dense0_bias"])
EXPECTED_MARCH_DENSE0_KERNEL = jnp.asarray(_MARCH_FIXTURE["dense0_kernel"])
EXPECTED_MARCH_DENSE1_BIAS   = jnp.asarray(_MARCH_FIXTURE["dense1_bias"])
EXPECTED_MARCH_DENSE1_KERNEL = jnp.asarray(_MARCH_FIXTURE["dense1_kernel"])
EXPECTED_MARCH_ORBITALS      = jnp.asarray(_MARCH_FIXTURE["orbitals"])


def test_march_updates_first_step():
    """Regression test: MARCH optimizer updates on the first training-loop
    iteration (after one sampling sweep), keys matching
    examples/hubbard_foundation_model/main.py."""
    lattice = square(shape=(L, L))

    Hs = [hubbard_square_pbc(L, U=U) for U in Us]
    H = combine_systems(Hs, N_mc_per_system)
    system_couplings = extract_system_couplings(H)
    system_ids = jnp.repeat(jnp.arange(n_systems), N_mc_per_system)

    initial_occupations = frozen_config("foundation_square16_ne10_nmc32")
    state = FermionFoundationState(
        occupations=initial_occupations,
        lattice=lattice,
        Ne=Ne,
        system_couplings=system_couplings,
        system_ids=system_ids,
        n_systems=n_systems,
    )

    model = FermionFoundationRBM(hidden_units=64)
    params = frozen_params("foundation_hidden64")
    wf = WaveFunction(params=params, apply_fn=model.apply, dtype=jnp.float64)

    action = BondExchange.create(lattice, max_dist=1, Nbands=2)

    optimizer = MARCH(diag_shift=1e-4, mode="real")
    opt_state = optimizer.init(wf.params)

    mc_keys = jax.random.split(jax.random.key(2), N_mc)
    state, log_amps, acceptance = sample(1, state, action, mc_keys, wf)
    E_L, e_mean, e2_mean = compute_expectation(H, wf, state, log_amps)
    updates, opt_state = optimizer(E_L, opt_state, state, wf)

    assert jnp.allclose(updates["params"]["Dense_0"]["bias"], EXPECTED_MARCH_DENSE0_BIAS, atol=1e-8)
    assert jnp.allclose(updates["params"]["Dense_0"]["kernel"], EXPECTED_MARCH_DENSE0_KERNEL, atol=1e-8)
    assert jnp.allclose(updates["params"]["Dense_1"]["bias"], EXPECTED_MARCH_DENSE1_BIAS, atol=1e-8)
    assert jnp.allclose(updates["params"]["Dense_1"]["kernel"], EXPECTED_MARCH_DENSE1_KERNEL, atol=1e-8)
    assert jnp.allclose(updates["params"]["orbitals"], EXPECTED_MARCH_ORBITALS, atol=1e-8)


_PARAMS_AFTER_5_STEPS_FIXTURE = np.load(DATA_DIR / "params_after_5_steps.npz")
EXPECTED_PARAMS_5_DENSE0_BIAS   = jnp.asarray(_PARAMS_AFTER_5_STEPS_FIXTURE["dense0_bias"])
EXPECTED_PARAMS_5_DENSE0_KERNEL = jnp.asarray(_PARAMS_AFTER_5_STEPS_FIXTURE["dense0_kernel"])
EXPECTED_PARAMS_5_DENSE1_BIAS   = jnp.asarray(_PARAMS_AFTER_5_STEPS_FIXTURE["dense1_bias"])
EXPECTED_PARAMS_5_DENSE1_KERNEL = jnp.asarray(_PARAMS_AFTER_5_STEPS_FIXTURE["dense1_kernel"])
EXPECTED_PARAMS_5_ORBITALS      = jnp.asarray(_PARAMS_AFTER_5_STEPS_FIXTURE["orbitals"])


def test_wf_params_after_5_training_steps():
    """Regression test: wf.params after 5 full training-loop iterations
    (sample -> compute_expectation -> MARCH update -> apply_gradients),
    keys matching examples/hubbard_foundation_model/main.py. Note mc_keys is
    re-derived from jax.random.key(2) on every iteration -- the same MC
    proposal keys are reused at every step, matching the script as written."""
    lattice = square(shape=(L, L))

    Hs = [hubbard_square_pbc(L, U=U) for U in Us]
    H = combine_systems(Hs, N_mc_per_system)
    system_couplings = extract_system_couplings(H)
    system_ids = jnp.repeat(jnp.arange(n_systems), N_mc_per_system)

    initial_occupations = frozen_config("foundation_square16_ne10_nmc32")
    state = FermionFoundationState(
        occupations=initial_occupations,
        lattice=lattice,
        Ne=Ne,
        system_couplings=system_couplings,
        system_ids=system_ids,
        n_systems=n_systems,
    )

    model = FermionFoundationRBM(hidden_units=64)
    params = frozen_params("foundation_hidden64")
    wf = WaveFunction(params=params, apply_fn=model.apply, dtype=jnp.float64)

    action = BondExchange.create(lattice, max_dist=1, Nbands=2)

    eta0 = 0.005
    lr_schedule = lambda step: eta0

    optimizer = MARCH(diag_shift=1e-4, mode="real")
    opt_state = optimizer.init(wf.params)

    N_steps = 5
    for step in range(N_steps):
        mc_keys = jax.random.split(jax.random.key(2), N_mc)
        state, log_amps, acceptance = sample(1, state, action, mc_keys, wf)
        E_L, e_mean, e2_mean = compute_expectation(H, wf, state, log_amps)
        lr = lr_schedule(step)
        updates, opt_state = optimizer(E_L, opt_state, state, wf)
        wf = wf.apply_gradients(updates, lr)

    assert jnp.allclose(wf.params["params"]["Dense_0"]["bias"], EXPECTED_PARAMS_5_DENSE0_BIAS, atol=1e-8)
    assert jnp.allclose(wf.params["params"]["Dense_0"]["kernel"], EXPECTED_PARAMS_5_DENSE0_KERNEL, atol=1e-8)
    assert jnp.allclose(wf.params["params"]["Dense_1"]["bias"], EXPECTED_PARAMS_5_DENSE1_BIAS, atol=1e-8)
    assert jnp.allclose(wf.params["params"]["Dense_1"]["kernel"], EXPECTED_PARAMS_5_DENSE1_KERNEL, atol=1e-8)
    assert jnp.allclose(wf.params["params"]["orbitals"], EXPECTED_PARAMS_5_ORBITALS, atol=1e-8)


_PARAMS_AFTER_5_STEPS_SR_FIXTURE = np.load(DATA_DIR / "params_after_5_steps_sr.npz")
EXPECTED_PARAMS_5_SR_DENSE0_BIAS   = jnp.asarray(_PARAMS_AFTER_5_STEPS_SR_FIXTURE["dense0_bias"])
EXPECTED_PARAMS_5_SR_DENSE0_KERNEL = jnp.asarray(_PARAMS_AFTER_5_STEPS_SR_FIXTURE["dense0_kernel"])
EXPECTED_PARAMS_5_SR_DENSE1_BIAS   = jnp.asarray(_PARAMS_AFTER_5_STEPS_SR_FIXTURE["dense1_bias"])
EXPECTED_PARAMS_5_SR_DENSE1_KERNEL = jnp.asarray(_PARAMS_AFTER_5_STEPS_SR_FIXTURE["dense1_kernel"])
EXPECTED_PARAMS_5_SR_ORBITALS      = jnp.asarray(_PARAMS_AFTER_5_STEPS_SR_FIXTURE["orbitals"])


def test_wf_params_after_5_training_steps_sr():
    """Same as test_wf_params_after_5_training_steps but with the SR optimizer
    (plain Stochastic Reconfiguration, no momentum/preconditioning) in place
    of MARCH."""
    lattice = square(shape=(L, L))

    Hs = [hubbard_square_pbc(L, U=U) for U in Us]
    H = combine_systems(Hs, N_mc_per_system)
    system_couplings = extract_system_couplings(H)
    system_ids = jnp.repeat(jnp.arange(n_systems), N_mc_per_system)

    initial_occupations = frozen_config("foundation_square16_ne10_nmc32")
    state = FermionFoundationState(
        occupations=initial_occupations,
        lattice=lattice,
        Ne=Ne,
        system_couplings=system_couplings,
        system_ids=system_ids,
        n_systems=n_systems,
    )

    model = FermionFoundationRBM(hidden_units=64)
    params = frozen_params("foundation_hidden64")
    wf = WaveFunction(params=params, apply_fn=model.apply, dtype=jnp.float64)

    action = BondExchange.create(lattice, max_dist=1, Nbands=2)

    eta0 = 0.005
    lr_schedule = lambda step: eta0

    optimizer = SR(diag_shift=1e-4, mode="real")
    opt_state = optimizer.init(wf.params)

    N_steps = 5
    for step in range(N_steps):
        mc_keys = jax.random.split(jax.random.key(2), N_mc)
        state, log_amps, acceptance = sample(1, state, action, mc_keys, wf)
        E_L, e_mean, e2_mean = compute_expectation(H, wf, state, log_amps)
        lr = lr_schedule(step)
        updates, opt_state = optimizer(E_L, opt_state, state, wf)
        wf = wf.apply_gradients(updates, lr)

    assert jnp.allclose(wf.params["params"]["Dense_0"]["bias"], EXPECTED_PARAMS_5_SR_DENSE0_BIAS, atol=1e-8)
    assert jnp.allclose(wf.params["params"]["Dense_0"]["kernel"], EXPECTED_PARAMS_5_SR_DENSE0_KERNEL, atol=1e-8)
    assert jnp.allclose(wf.params["params"]["Dense_1"]["bias"], EXPECTED_PARAMS_5_SR_DENSE1_BIAS, atol=1e-8)
    assert jnp.allclose(wf.params["params"]["Dense_1"]["kernel"], EXPECTED_PARAMS_5_SR_DENSE1_KERNEL, atol=1e-8)
    assert jnp.allclose(wf.params["params"]["orbitals"], EXPECTED_PARAMS_5_SR_ORBITALS, atol=1e-8)
