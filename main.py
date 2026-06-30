from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.experimental.shard_map import shard_map

from tachys.lattice.spins.hamiltonians.heisenberg import heisenberg_square_pbc
from tachys.lattice.spins.hamiltonians.ising_transverse_field import ising_transverse_field_square_pbc
from tachys.lattice.spins.spin_state import SpinState, init_config_fixed_magn
from tachys.lattice.spins.spin_action import SpinFlip
from tachys.montecarlo import CompositeAction
from tachys.lattice.ansatz.rbm import SpinRBM
from tachys.lattice.operator.local_estimator import compute_expectation, local_estimator
from tachys.montecarlo import sample
from tachys.optimizer.optimizers import _build_ntk as _build_ntk_base
from tachys.parallel import mesh
from tachys.wavefunction import WaveFunction
from tachys.optimizer import SR, SPRING, MARCH

L = 4
N = L * L
N_mc = 16
N_hidden = N  # alpha=1 hidden units

# Heisenberg Hamiltonian on a 4x4 square lattice with PBC
H = ising_transverse_field_square_pbc(L)

# RBM wavefunction
model = SpinRBM(num_hidden=1, dtype=jnp.float64, complex=False)

# Batch of 16 zero-magnetisation spin configurations
spins = init_config_fixed_magn(jax.random.key(1), N, sz=0, N_mc=N_mc)
state = SpinState(spins=spins, Ns=N)

# Initialise RBM parameters with a dummy forward pass
params = model.init(jax.random.key(0), state)
wf = WaveFunction(params=params, apply_fn=model.apply)

# MC sampling
action = SpinFlip()

N_steps = 5
eta     = 0.01

optimizer = SR(diag_shift=1e-4, mode="real")
opt_state = optimizer.init(wf.params)

import time

class Timer:
    def __enter__(self):
        self._t0 = time.perf_counter()
        return self
    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._t0

print("\n--- SR optimization ---")
for step in range(N_steps):
    mc_keys = jax.random.split(jax.random.key(2), N_mc)

    with Timer() as t_sample:
        state, log_amps, acceptance = sample(1, state, action, mc_keys, wf)
        jax.block_until_ready((state, log_amps))

    with Timer() as t_expect:
        E_L, e_mean, _ = compute_expectation(H, wf, state, log_amps)
        jax.block_until_ready(E_L)

    with Timer() as t_opt:
        updates, opt_state = optimizer(E_L, opt_state, state, wf)
        jax.block_until_ready(updates)
        wf = wf.apply_gradients(updates, eta)

    print(f"  step {step:2d}  E/N = {e_mean / N:.6f}  "
          f"sample={t_sample.elapsed*1e3:.1f}ms  "
          f"expect={t_expect.elapsed*1e3:.1f}ms  "
          f"opt={t_opt.elapsed*1e3:.1f}ms")

print(wf.params)