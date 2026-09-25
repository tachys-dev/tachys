"""Integration tests: 4-process MPI SR loops match single-process reference values.

Each test in this module calls the module-scoped ``parallel_results`` fixture,
which launches

    JAX_PLATFORMS=cpu mpirun --oversubscribe -np 4 \\
        python tests/parallel/run_sr_loop.py <tmpfile>

once for the whole module and parses the JSON written by rank 0.
The expected energies and final parameters are identical to those used in
tests/optimizer/test_sr.py, confirming that the 4-process distributed run
is numerically equivalent to the serial run. The same loops rerun with the
optimizer's ``dtype=float64`` must match the default ones.

The entire module is skipped automatically when ``mpirun`` is not found on PATH.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import jax.numpy as jnp
import pytest

_RUNNER = Path(__file__).parent / "run_sr_loop.py"


@pytest.fixture(scope="module")
def mpirun_bin():
    path = shutil.which("mpirun")
    if path is None:
        pytest.skip("mpirun not found on PATH — skipping MPI parallel tests")
    return path


@pytest.fixture(scope="module")
def parallel_results(mpirun_bin):
    """Run the SR loop with 4 MPI processes; return parsed JSON results."""
    fd, output_path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        env = os.environ.copy()
        env["JAX_PLATFORMS"] = "cpu"

        proc = subprocess.run(
            [
                mpirun_bin,
                "--oversubscribe",   # allow >1 process per physical core
                "-np", "4",
                sys.executable, str(_RUNNER), output_path,
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )

        if proc.returncode != 0:
            pytest.fail(
                f"mpirun exited with code {proc.returncode}\n"
                f"--- stdout ---\n{proc.stdout}\n"
                f"--- stderr ---\n{proc.stderr}"
            )

        with open(output_path) as f:
            return json.load(f)
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)


# ---------------------------------------------------------------------------
# Complex mode (matches test_sr_loop_5_steps in tests/optimizer/test_sr.py)
# ---------------------------------------------------------------------------

_EXPECTED_ENERGIES_COMPLEX = [
    -0.99646772696840,
    -0.92262547725766,
    -1.07355544444143,
    -1.03770217010731,
    -0.98556413159589,
]

_EXPECTED_PARAMS_COMPLEX = dict(
    imag_linear_bias   = jnp.array([-0.02153195]),
    imag_linear_kernel = jnp.array([[-0.16989735]]),
    linear_bias        = jnp.array([-0.09049892]),
    linear_kernel      = jnp.array([
        [-0.00549538], [-0.57975204], [-0.42330984], [ 0.30260539],
        [ 0.42412032], [-0.26708997], [-0.02215112], [ 0.64839192],
        [ 0.0194746 ], [-0.621071  ], [-0.17652029], [ 0.06760028],
        [-0.09426484], [ 0.56902585], [ 0.48198705], [ 0.41160606],
    ]),
)


@pytest.mark.mpi
def test_complex_mode_energies(parallel_results):
    """5-step complex-mode energies match the single-process reference."""
    energies = parallel_results["complex"]["energies"]
    assert len(energies) == 5
    for step, (got, exp) in enumerate(zip(energies, _EXPECTED_ENERGIES_COMPLEX)):
        assert abs(got - exp) < 1e-12, (
            f"complex step {step}: expected {exp:.14f}, got {got:.14f}"
        )


@pytest.mark.mpi
def test_complex_mode_final_params(parallel_results):
    """Final complex-mode parameters match the single-process reference."""
    p = parallel_results["complex"]["final_params"]
    for key, expected in _EXPECTED_PARAMS_COMPLEX.items():
        got = jnp.array(p[key])
        assert jnp.allclose(got, expected, atol=1e-7), (
            f"complex {key}: max deviation {float(jnp.max(jnp.abs(got - expected))):.2e}"
        )


# ---------------------------------------------------------------------------
# Real mode (matches test_sr_real_loop_5_steps in tests/optimizer/test_sr.py)
# ---------------------------------------------------------------------------

_EXPECTED_ENERGIES_REAL = [
    -1.00735553414256,
    -0.93472240035012,
    -1.12213451032061,
    -0.96788517098408,
    -1.10386617541288,
]

_EXPECTED_PARAMS_REAL = dict(
    linear_bias   = jnp.array([-0.45330682]),
    linear_kernel = jnp.array([
        [-0.37486956], [-0.4709006 ], [-0.07152381], [ 0.09834236],
        [ 0.34400194], [-0.25285732], [-0.05991222], [ 0.27908684],
        [ 0.06448295], [-0.17423916], [ 0.71773061], [ 0.16435237],
        [-0.06640457], [ 0.2647537 ], [ 0.43440253], [ 0.3993689 ],
    ]),
)


@pytest.mark.mpi
def test_real_mode_energies(parallel_results):
    """5-step real-mode energies match the single-process reference."""
    energies = parallel_results["real"]["energies"]
    assert len(energies) == 5
    for step, (got, exp) in enumerate(zip(energies, _EXPECTED_ENERGIES_REAL)):
        assert abs(got - exp) < 1e-12, (
            f"real step {step}: expected {exp:.14f}, got {got:.14f}"
        )


@pytest.mark.mpi
def test_real_mode_final_params(parallel_results):
    """Final real-mode parameters match the single-process reference."""
    p = parallel_results["real"]["final_params"]
    for key, expected in _EXPECTED_PARAMS_REAL.items():
        got = jnp.array(p[key])
        assert jnp.allclose(got, expected, atol=1e-7), (
            f"real {key}: max deviation {float(jnp.max(jnp.abs(got - expected))):.2e}"
        )


# ---------------------------------------------------------------------------
# Optimizer dtype (tachys.optimizer: the shifted-Jacobian path)
# ---------------------------------------------------------------------------

@pytest.mark.mpi
@pytest.mark.parametrize("mode", ["complex", "real"])
def test_float64_dtype_matches_the_default_loop(parallel_results, mode):
    """With an optimizer dtype, every Jacobian is shifted by a pmean'd mean
    before the contraction. The centering removes the shift only if all
    processes use the same one: with per-process means instead, these loops
    part by 0.1 in energy within the 5 steps."""
    ref, got = parallel_results[mode], parallel_results[f"{mode}_float64"]
    for step, (e, e_ref) in enumerate(zip(got["energies"], ref["energies"])):
        assert abs(e - e_ref) < 1e-12, f"{mode} step {step}: {e:.14f} vs {e_ref:.14f}"
    for key, expected in ref["final_params"].items():
        diff = jnp.max(jnp.abs(jnp.array(got["final_params"][key]) - jnp.array(expected)))
        assert diff < 1e-10, f"{mode} {key}: max deviation {float(diff):.2e}"
