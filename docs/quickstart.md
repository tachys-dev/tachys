# Quickstart

Variational Monte Carlo (VMC) minimizes the energy of a variational
wavefunction: configurations are sampled from $|\psi|^2$, the energy is
estimated on that sample, and the parameters are updated to lower it.

This page gives two complete scripts — the $J_1$-$J_2$ Heisenberg model
(spins) and the Hubbard model (fermions), both on a 4×4 square lattice. They
differ in the Hamiltonian, the state and the ansatz; the optimization loop is
identical. Each step prints the energy per site.

## Structure of a VMC script

Both scripts are built from the same five components:

1. **Lattice and Hamiltonian.** A lattice factory and an operator factory; the
   Hamiltonian is a Python callable.
2. **Initial configurations.** One per Markov chain, drawn at random within
   the conserved sector — total magnetization for spins, particle number for
   fermions.
3. **Wavefunction.** Any Flax module; the examples use the transformer
   ansätze included in the library. `WaveFunction` pairs the parameters with
   the module's `apply` function, which makes the rest of tachys independent
   of the architecture.
4. **Monte Carlo move.** A rule for proposing a new configuration from the
   current one, here the exchange of two sites, defining a Markov chain that
   samples configurations with probability $|\psi|^2$.
5. **Optimization step.** A map from the sampled energies to a parameter
   update. Both examples use stochastic reconfiguration (`SR`), which takes a
   natural-gradient step in place of a plain gradient step.

One step of the loop is four lines: advance the chain, estimate the energy,
solve for the update, apply it.

## Part 1 — Spins: a frustrated antiferromagnet

The Heisenberg model on a 4×4 square lattice, with an antiferromagnetic
nearest-neighbour coupling $J_1$ and a weaker next-nearest (diagonal) coupling
$J_2$:

$$
H = \sum_{\langle i,j \rangle} J_{ij}\; \mathbf{S}_i \cdot \mathbf{S}_j
$$

with $J_{ij} = J_1$ on nearest-neighbour bonds and $J_{ij} = J_2$ on the
diagonal ones. The $J_2$ term frustrates the Néel order favoured by $J_1$
alone.

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

The same five components, with occupation numbers in place of spins and a
determinant-based ansatz. The Hubbard model at half filling — one electron per
site on average — on the same 4×4 square lattice:

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

- {doc}`concepts` — the data types and functions used above.
- {doc}`foundation_models` — one wavefunction trained across many Hamiltonians.
- {doc}`api` — full reference: ansätze, optimizers, and the lower-level
  interfaces.
