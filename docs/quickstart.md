# Quickstart

Tachys is a library for **variational Monte Carlo (VMC)**: you pick a Hamiltonian,
pick a neural network to represent the wavefunction, and optimize that network so
its energy gets as low as possible. Concretely, that means repeating three things
in a loop: *sample* configurations from the current wavefunction, *estimate* the
energy on that sample, and *update* the network's parameters to lower it.

This page walks through that loop twice, end to end and with nothing hidden: once
for a **spin** model (an antiferromagnet on a square lattice) and once for a
**fermionic** model (the Hubbard model). Copy either example in and run it — the
printed energy per site will drop step by step.

## The shape of every tachys VMC script

Both examples below follow the same five pieces:

1. **A lattice and a Hamiltonian.** A lattice factory plus an operator factory —
   the Hamiltonian ends up being just a Python callable.
2. **A batch of starting configurations.** One configuration per Markov chain,
   sampled randomly subject to whatever's conserved (total magnetization for
   spins, particle number for fermions).
3. **A wavefunction.** Any Flax neural network can be a tachys wavefunction —
   here we use the transformer-based ansätze that ship with the library. Wrapping
   the network's parameters and its `apply` function together in a
   `WaveFunction` is what lets the rest of tachys treat any architecture the
   same way.
4. **A Monte Carlo move.** A rule for proposing a new configuration from the
   current one (e.g. "swap two sites"), used to run a Markov chain that samples
   configurations with probability $|\psi|^2$.
5. **An optimization step.** Turn the sampled energies into a parameter update.
   Here we use stochastic reconfiguration (`SR`), which takes a natural-gradient
   step rather than a plain gradient step.

With those five pieces in hand, one optimization step is only four lines:
advance the Markov chain, estimate the energy, solve for the update, apply it.

## Part 1 — Spins: a frustrated antiferromagnet

Our first example is the Heisenberg model on a 4×4 square lattice, with an
antiferromagnetic nearest-neighbour coupling $J_1$ and a weaker next-nearest
(diagonal) coupling $J_2$ that *frustrates* the simple checkerboard order $J_1$
alone would favor:

$$
H = \sum_{\langle i,j \rangle} J_{ij}\; \mathbf{S}_i \cdot \mathbf{S}_j
$$

with $J_{ij} = J_1$ on nearest-neighbour bonds and $J_{ij} = J_2$ on the
frustrating next-nearest (diagonal) bonds.

```python
import jax
import jax.numpy as jnp

from tachys.lattice.lattice_database import square
from tachys.lattice.spins.spin_state import SpinState, init_config_fixed_magn
from tachys.lattice.spins.hamiltonians.heisenberg import heisenberg_hamiltonian
from tachys.lattice.ansatz.spin_vit import SpinViT
from tachys.lattice.bond_exchange import BondExchange
from tachys.wavefunction import WaveFunction
from tachys.optimizer import SR
from tachys.montecarlo import sample
from tachys.lattice.operator.local_estimator import compute_expectation

L = 4
lattice = square(shape=(L, L))
N = lattice.Ns

J1, J2 = 1.0, 0.5
H = heisenberg_hamiltonian(lattice, nn=[
    ((1, 0), J1), ((0, 1), J1),
    ((1, 1), J2), ((1, -1), J2),
])

key = jax.random.key(0)
N_mc = 1024
key, subkey = jax.random.split(key)
spins = init_config_fixed_magn(subkey, N, N_mc=N_mc)
state = SpinState(spins=spins, lattice=lattice)

model = SpinViT(num_layers=1, d_model=16, num_heads=4, seq_len=N, b=1,
                transl_invariant=True, two_dimensional=True)
key, subkey = jax.random.split(key)
params = model.init(subkey, state)
wf = WaveFunction(params=params, apply_fn=model.apply)

action = BondExchange.create(lattice)
optimizer = SR(diag_shift=1e-3, mode="complex")
opt_state = optimizer.init(wf.params)

N_steps, lr = 30, 5e-2
for step in range(N_steps):
    key, subkey = jax.random.split(key)
    mc_keys = jax.random.split(subkey, N_mc)

    state, log_amps, acceptance = sample(1, state, action, mc_keys, wf)
    E_L, e_mean, e2_mean = compute_expectation(H, wf, state, log_amps)

    updates, opt_state = optimizer(E_L, opt_state, state, wf)
    wf = wf.apply_gradients(updates, lr)

    print(f"step {step:3d}  E/N = {jnp.real(e_mean) / N: .4f}")
```

## Part 2 — Fermions: the Hubbard model

The fermionic side follows the same five steps, with the state tracking
occupation numbers instead of spins and the wavefunction built as a Slater
determinant. We'll use the Hubbard model at half filling (one electron per
site on average) on the same 4×4 square lattice:

$$
H = -t \sum_{\langle i,j \rangle, \sigma} (c^\dagger_{i\sigma} c_{j\sigma} + \text{h.c.})
    + U \sum_i n_{i\uparrow} n_{i\downarrow}
$$

```python
import jax
import jax.numpy as jnp

from tachys.lattice.lattice_database import square
from tachys.lattice.fermions.fermion_state import FermionState, init_config_spinful
from tachys.lattice.fermions.hamiltonians.hubbard import hubbard_hamiltonian
from tachys.lattice.ansatz.fermionic_transformer import FermionicTransformer
from tachys.lattice.bond_exchange import BondExchange
from tachys.wavefunction import WaveFunction
from tachys.optimizer import SR
from tachys.montecarlo import sample
from tachys.lattice.operator.local_estimator import compute_expectation

L = 4
lattice = square(shape=(L, L))
N = lattice.Ns

t, U = 1.0, 4.0
H = hubbard_hamiltonian(lattice, nn=[((1, 0), t), ((0, 1), t)], U=U)

Ne = N  # half filling
key = jax.random.key(0)
N_mc = 1024
key, subkey = jax.random.split(key)
occupations, N_up, N_down = init_config_spinful(subkey, Ns=N, Ne=Ne, N_mc=N_mc)
state = FermionState(occupations=occupations, Ne=Ne, lattice=lattice)

model = FermionicTransformer(num_layers=1, d_model=16, num_heads=4, Ne=Ne, Ns=N)
key, subkey = jax.random.split(key)
params = model.init(subkey, state)
wf = WaveFunction(params=params, apply_fn=model.apply)

action = BondExchange.create(lattice, Nbands=2)
optimizer = SR(diag_shift=1e-3, mode="complex")
opt_state = optimizer.init(wf.params)

N_steps, lr = 30, 5e-2
for step in range(N_steps):
    key, subkey = jax.random.split(key)
    mc_keys = jax.random.split(subkey, N_mc)

    state, log_amps, acceptance = sample(1, state, action, mc_keys, wf)
    E_L, e_mean, e2_mean = compute_expectation(H, wf, state, log_amps)

    updates, opt_state = optimizer(E_L, opt_state, state, wf)
    wf = wf.apply_gradients(updates, lr)

    print(f"step {step:3d}  E/N = {jnp.real(e_mean) / N: .4f}")
```

## Next steps

- {doc}`concepts` — states, operators, and the algebra behind them, in depth.
- {doc}`foundation_models` — train a single wavefunction across many
  Hamiltonians at once.
- {doc}`api` — the full reference: every wavefunction ansatz, every optimizer,
  and the lower-level pieces this page skipped over.
