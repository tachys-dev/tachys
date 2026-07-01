"""Runner script for parallel SR regression tests.

Runs the same 5-step SR loops as test_sr_loop_5_steps and test_sr_real_loop_5_steps
(both complex and real modes) under the current JAX device configuration.  Results
are written as JSON to the path given as the first command-line argument; only
rank 0 writes the file.

Typical invocation from the project root:

    JAX_PLATFORMS=cpu mpirun --oversubscribe -np 4 \\
        python tests/parallel/run_sr_loop.py /tmp/sr_results.json
"""
import json
import sys
from pathlib import Path

# Ensure the project root is on sys.path regardless of working directory.
# When running as a script (not via -c), Python sets sys.path[0] to the
# script's own directory; we need the repo root so that `import tachys` works
# even when tachys has not been pip-installed.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import jax
import jax.numpy as jnp

# Importing any tachys sub-module triggers tachys/__init__.py, which calls
# jax.distributed.initialize() when the MPI environment is detected and
# enables 64-bit floats.
from tachys.lattice.ansatz.rbm import SpinRBM
from tachys.lattice.operator.local_estimator import compute_expectation
from tachys.lattice.spins.hamiltonians.ising_transverse_field import (
    ising_transverse_field_square_pbc,
)
from tachys.lattice.spins.spin_action import SpinFlip
from tachys.lattice.spins.spin_state import SpinState, init_config_fixed_magn
from tachys.lattice.lattice_database import square
from tachys.montecarlo import sample
from tachys.optimizer import SR
from tachys.parallel import rank
from tachys.wavefunction import WaveFunction

L = 4
N = L * L
N_mc = 16


def _run_loop(complex_mode: bool) -> dict:
    mode = "complex" if complex_mode else "real"
    H = ising_transverse_field_square_pbc(L, J=1.0, h=1.0)
    model = SpinRBM(num_hidden=1, dtype=jnp.float64, complex=complex_mode)

    lattice = square(shape=(L, L))
    spins = init_config_fixed_magn(jax.random.key(1), N, sz=0, N_mc=N_mc)
    state = SpinState(spins=spins, lattice=lattice)
    params = model.init(jax.random.key(0), state)
    wf = WaveFunction(params=params, apply_fn=model.apply)

    action = SpinFlip()
    eta = 0.01
    optimizer = SR(diag_shift=1e-4, mode=mode)
    opt_state = optimizer.init(wf.params)

    energies = []
    for _ in range(5):
        mc_keys = jax.random.split(jax.random.key(2), N_mc)
        state, log_amps, _ = sample(1, state, action, mc_keys, wf)
        E_L, e_mean, _ = compute_expectation(H, wf, state, log_amps)
        updates, opt_state = optimizer(E_L, opt_state, state, wf)
        wf = wf.apply_gradients(updates, eta)
        energies.append(float(e_mean.real) / N)

    p = wf.params["params"]
    if complex_mode:
        final_params = {
            "imag_linear_bias":   p["imag_linear"]["bias"].tolist(),
            "imag_linear_kernel": p["imag_linear"]["kernel"].tolist(),
            "linear_bias":        p["linear"]["bias"].tolist(),
            "linear_kernel":      p["linear"]["kernel"].tolist(),
        }
    else:
        final_params = {
            "linear_bias":   p["linear"]["bias"].tolist(),
            "linear_kernel": p["linear"]["kernel"].tolist(),
        }

    return {"energies": energies, "final_params": final_params}


output_path = sys.argv[1]

results = {
    "complex": _run_loop(complex_mode=True),
    "real":    _run_loop(complex_mode=False),
}

if rank == 0:
    with open(output_path, "w") as f:
        json.dump(results, f)
