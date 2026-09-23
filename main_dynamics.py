
import jax
import jax.numpy as jnp
import numpy as np

from tachys.dynamics import TDVP, evolve
from tachys.experimental.fidelity import log_amplitudes
from tachys.ground_state_training import train
from tachys.lattice.ansatz.rbm import SpinRBM
from tachys.lattice.exact_diag import build_sparse_hamiltonian, spins_hilbert_space
from tachys.lattice.lattice_database import square
from tachys.lattice.operator.local_estimator import compute_expectation
from tachys.lattice.spins.hamiltonians.ising_transverse_field import (
    ising_transverse_field_square_pbc,
)
from tachys.lattice.spins.spin_action import SpinFlip
from tachys.lattice.spins.spin_operators import Sx
from tachys.lattice.spins.spin_state import SpinState, init_config_fixed_magn
from tachys.optimizer import MARCH
from tachys.parallel import MASTER, rank
from tachys.wavefunction import WaveFunction

L = 4
N = L * L
N_mc = 512
seed = 0

h0 = 2.0          # pre-quench transverse field (paramagnetic)
h1 = 1.0          # post-quench field
J = 1.0

N_opt_steps = 200
eta = 0.005

dt = 0.001
N_time_steps = 500

key = jax.random.key(seed)
lattice = square(shape=(L, L))

key, subkey = jax.random.split(key)
state = SpinState(spins=init_config_fixed_magn(subkey, N, sz=0, N_mc=N_mc), lattice=lattice)

model = SpinRBM(hidden_units=2 * N, dtype=jnp.float64, complex=True)
key, subkey = jax.random.split(key)
params = jax.tree.map(
    lambda x: 0.001 * jax.random.normal(
        jax.random.PRNGKey(42),
        x.shape,
        dtype=x.dtype,
    ),
    model.init(subkey, state),
)
wf = WaveFunction(params=params, apply_fn=model.apply, dtype=jnp.float64)

action = SpinFlip()

# ─── 1. Ground state of H(h0) ─────────────────────────────────────────────────

H_initial = ising_transverse_field_square_pbc(L, J=J, h=h0)

key, subkey = jax.random.split(key)
key, state, wf, _opt_state, _history = train(
    subkey, H_initial, state, wf, MARCH(diag_shift=1e-4, mode="complex"), action,
    N_opt_steps, lambda step: eta, N_mc,
)

# ─── 2. Quench to H(h1) and evolve ────────────────────────────────────────────

H_quench = ising_transverse_field_square_pbc(L, J=J, h=h1)

# diag_shift=0.0: the eigenvalue truncation below is the regularizer. rcond caps
# the condition number of the retained subspace at 1/rcond
tdvp = TDVP(mode="complex", rcond=1e-8)

# Transverse magnetization m_x = (1/N) sum_i sigma^x_i, with sigma^x = 2 S^x.
Mx = (2.0 / N) * Sx(tuple(range(N)))
mx_vmc = []

def measure_mx(state, wf, step):
    """Measure m_x on stage 1's batch, so it costs no extra sampling.

    ``evolve`` hands callbacks the wavefunction at the start of the step and the
    configurations sampled from it, i.e. the ones that belong with
    ``history["t"][step]``.
    """
    mx = compute_expectation(Mx, wf, state, log_amplitudes(wf, state))[1]
    mx_vmc.append(float(mx.real))
    return {"mx": mx_vmc[-1]}

key, subkey = jax.random.split(key)
key, state, wf, _opt_state, history = evolve(
    subkey, H_quench, state, wf, tdvp, action,
    N_steps=N_time_steps, dt=dt, N_mc=N_mc,
    integrator="heun",
    tdvp_error_every=10,
    log_callback_fn=measure_mx,
)