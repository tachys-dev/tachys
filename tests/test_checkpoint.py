import jax
import jax.numpy as jnp
import numpy as np
import pytest

wandb = pytest.importorskip("wandb")

from tachys.checkpoint import load_checkpoint
from tachys.ground_state_training import train
from tachys.lattice.ansatz.rbm import SpinRBM
from tachys.lattice.lattice_database import square
from tachys.lattice.spins.hamiltonians.ising_transverse_field import ising_transverse_field_square_pbc
from tachys.lattice.spins.spin_action import SpinFlip
from tachys.lattice.spins.spin_state import SpinState, init_config_fixed_magn
from tachys.optimizer import MARCH
from tachys.wavefunction import WaveFunction

L = 4
N = L * L
N_mc = 16
ETA = 0.01


def _build_problem():
    H = ising_transverse_field_square_pbc(L, J=1.0, h=1.0)
    model = SpinRBM(hidden_units=1, dtype=jnp.float64, complex=False)
    lattice = square(shape=(L, L))
    spins = init_config_fixed_magn(jax.random.key(1), N, sz=0, N_mc=N_mc)
    state = SpinState(spins=spins, lattice=lattice)
    params = model.init(jax.random.key(0), state)
    wf = WaveFunction(params=params, apply_fn=model.apply)
    action = SpinFlip()
    optimizer = MARCH(diag_shift=1e-4, mode="real")
    return H, state, wf, action, optimizer


def _lr_schedule(step):
    return ETA


def test_checkpoint_resume_matches_continuous_training(tmp_path, monkeypatch):
    """10 continuous steps must match 5 steps + checkpoint save/restore + 5 more steps."""
    # Sentry's own app_url() call triggers a DeprecationWarning inside wandb itself;
    # this is unrelated to what we're testing, so skip its error-reporting client entirely.
    monkeypatch.setenv("WANDB_ERROR_REPORTING", "false")

    start_key = jax.random.key(2)

    # Continuous 10-step run.
    H, state, wf, action, optimizer = _build_problem()
    _, _, wf_continuous, _, _ = train(
        start_key, H, state, wf, optimizer, action, 10, _lr_schedule, N_mc,
    )

    # First half: 5 steps, checkpointed every step via wandb (offline, local only).
    H, state, wf, action, optimizer = _build_problem()
    wandb_run = wandb.init(
        project="tachys-test", dir=str(tmp_path), mode="offline",
        config={"checkpoint_every": 1, "checkpoint_keep": 1},
    )
    key, state, wf, opt_state, _ = train(
        start_key, H, state, wf, optimizer, action, 5, _lr_schedule, N_mc, wandb_run=wandb_run,
    )
    ckpt_dir = f"{wandb_run.dir}/checkpoints"
    wandb_run.finish()

    # Restore from disk, then run the remaining 5 steps.
    opt_state_template = optimizer.init(wf.params)
    params, opt_state, state, key = load_checkpoint(
        ckpt_dir, state, opt_state_template, params_template=wf.params,
    )
    wf = wf.replace(params=params)
    _, _, wf_resumed, _, _ = train(
        key, H, state, wf, optimizer, action, 5, _lr_schedule, N_mc, opt_state=opt_state,
    )

    jax.tree_util.tree_map(
        lambda a, b: np.testing.assert_array_equal(a, b),
        wf_continuous.params, wf_resumed.params,
    )


def test_load_checkpoint_without_params_template(tmp_path, monkeypatch):
    """Omitting params_template should still recover correct params, opt_state, state and key."""
    monkeypatch.setenv("WANDB_ERROR_REPORTING", "false")

    H, state, wf, action, optimizer = _build_problem()
    wandb_run = wandb.init(
        project="tachys-test", dir=str(tmp_path), mode="offline",
        config={"checkpoint_every": 1, "checkpoint_keep": 1},
    )
    key, state, wf, opt_state, _ = train(
        jax.random.key(2), H, state, wf, optimizer, action, 2, _lr_schedule, N_mc, wandb_run=wandb_run,
    )
    ckpt_dir = f"{wandb_run.dir}/checkpoints"
    wandb_run.finish()

    opt_state_template = optimizer.init(wf.params)
    params, restored_opt_state, restored_state, restored_key = load_checkpoint(
        ckpt_dir, state, opt_state_template,
    )

    jax.tree_util.tree_map(lambda a, b: np.testing.assert_array_equal(a, b), wf.params, params)
    jax.tree_util.tree_map(lambda a, b: np.testing.assert_array_equal(a, b), opt_state, restored_opt_state)
    np.testing.assert_array_equal(state.config, restored_state.config)
    np.testing.assert_array_equal(jax.random.key_data(key), jax.random.key_data(restored_key))
