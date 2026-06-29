import jax
import jax.numpy as jnp

from tachys.lattice.spins.hamiltonians.heisenberg import heisenberg_square_pbc
from tachys.lattice.spins.spin_state import SpinState, init_config_fixed_magn
from tachys.lattice.spins.spin_action import SpinFlip
from tachys.lattice.ansatz.rbm import SpinRBM
from tachys.lattice.operator.local_estimator import local_estimator
from tachys.montecarlo import sample
from tachys.wavefunction import WaveFunction

L = 4
N = L * L
N_mc = 16
N_hidden = N  # alpha=1 hidden units

key = jax.random.key(0)

# Heisenberg Hamiltonian on a 4x4 square lattice with PBC
H = heisenberg_square_pbc(L, J=1.0)

# RBM wavefunction
model = SpinRBM(num_hidden=N_hidden, dtype=jnp.float64)

# Batch of 16 zero-magnetisation spin configurations
key, subkey = jax.random.split(key)
spins = init_config_fixed_magn(subkey, N, sz=0, N_mc=N_mc)
state = SpinState(spins=spins, Ns=N)

# Initialise RBM parameters with a dummy forward pass
key, subkey = jax.random.split(key)
dummy = SpinState(spins=jnp.ones((1, N), dtype=jnp.float64), Ns=N)
params = model.init(subkey, dummy)
wf = WaveFunction(params=params, apply_fn=model.apply)

# Log-amplitudes for the batch
log_amp = wf.apply_fn(wf.params, state)

# Local energy O_L(x) = sum_{x'} <x|H|x'> psi(x') / psi(x)
O_L = local_estimator(H, state, wf, log_amp, optimize_mask=False)

print("configurations shape:", spins.shape)
print("log amplitudes shape:", log_amp.shape)
print("local energy shape:  ", O_L.shape)
print("local energies:\n", O_L)
print("mean local energy:", jnp.mean(O_L))

# MC sampling
action = SpinFlip()
key, subkey = jax.random.split(key)
mc_keys = jax.random.split(subkey, N_mc)

state, log_amps, acceptance = sample(10, state, action, mc_keys, wf)

print("\nafter 10 sweeps:")
print("acceptance rate:", acceptance)
print("log_amps shape: ", log_amps.shape)

