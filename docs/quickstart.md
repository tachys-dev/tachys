# Quickstart

Variational Monte Carlo (VMC) approximates the ground state of a Hamiltonian
by minimizing the energy of a variational wavefunction $\psi$. Each step
samples configurations from $|\psi|^2$, estimates the energy on that sample,
and updates the parameters to lower it.

This page gives two complete scripts on a $4 \times 4$ square lattice: the
$J_1$-$J_2$ Heisenberg model, for spins, and the Hubbard model, for fermions.
They differ in the Hamiltonian, the configurations and the ansatz; the
optimization loop is the same.

## Structure of a VMC script

A script is built from five components, each covered on its own page of the
{doc}`guide/index`:

1. **{doc}`Lattice <guide/lattice>` and
   {doc}`Hamiltonian <guide/hamiltonian>`.** The lattice gives the sites and
   the bonds; the Hamiltonian is a sum of spin or fermionic operators on them.
2. **{doc}`Initial configuration <guide/configuration>`.** One per Markov
   chain, drawn at random with the conserved quantum numbers fixed: the total
   magnetization for spins, the number of electrons of each spin for fermions.
3. **{doc}`Wavefunction <guide/wavefunction>`.** Any Flax module that returns
   $\log\psi$; the scripts use transformer ansätze of the library.
   `WaveFunction` pairs its parameters with its `apply` function.
4. **{doc}`Monte Carlo move <guide/sampling>`.** The rule that proposes a new
   configuration from the current one: here, the exchange of two neighbouring
   spins, or the hop of an electron to a neighbouring site.
5. **{doc}`Optimization step <guide/optimization>`.** The map from the local
   energies to a parameter update: here, stochastic reconfiguration (`SR`), a
   natural-gradient method.

A step of the loop is four lines: advance the chains, estimate the energy,
compute the update, apply it.

## Spins: the $J_1$-$J_2$ Heisenberg model

$$
H = J_1 \sum_{\langle i,j \rangle} \mathbf{S}_i \cdot \mathbf{S}_j
  + J_2 \sum_{\langle\langle i,j \rangle\rangle} \mathbf{S}_i \cdot \mathbf{S}_j ,
$$

where $\langle i,j \rangle$ runs over nearest neighbours and
$\langle\langle i,j \rangle\rangle$ over next-nearest neighbours, across the
diagonals of the squares. The $J_2$ term frustrates the Néel order favoured by
$J_1$ alone.

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

# 1. Lattice and Hamiltonian
L = 4
lattice = square(shape=(L, L))
N = lattice.Ns

J1, J2 = 1.0, 0.5
H = heisenberg_hamiltonian(lattice, nn=[
    ((1, 0), J1), ((0, 1), J1),         # nearest neighbours
    ((1, 1), J2), ((1, -1), J2),        # next-nearest neighbours
])

# 2. Initial configurations: N_mc chains, total magnetization 0
key = jax.random.key(0)
N_mc = 1024
key, subkey = jax.random.split(key)
spins = init_config_fixed_magn(subkey, N, N_mc=N_mc)
state = SpinState(spins=spins, lattice=lattice)

# 3. Wavefunction
model = SpinViT(num_layers=1, d_model=16, num_heads=4, seq_len=N, b=1,
                transl_invariant=True, two_dimensional=True)
key, subkey = jax.random.split(key)
params = model.init(subkey, state)
wf = WaveFunction(params=params, apply_fn=model.apply)

# 4. Monte Carlo move
action = BondExchange.create(lattice)

# 5. Optimizer
optimizer = SR(diag_shift=1e-4, mode="complex")
opt_state = optimizer.init(wf.params)

N_steps, nsweeps, lr = 30, 1, 0.03
for step in range(N_steps):
    key, subkey = jax.random.split(key)
    mc_keys = jax.random.split(subkey, N_mc)

    state, log_amps, acceptance = sample(nsweeps, state, action, mc_keys, wf)
    E_L, e_mean, e2_mean = compute_expectation(H, wf, state, log_amps)

    updates, opt_state = optimizer(E_L, opt_state, state, wf)
    wf = wf.apply_gradients(updates, lr)

    print(f"step {step:3d}  E/N = {jnp.real(e_mean) / N: .4f}")
```

At every step, `sample` advances each chain by `nsweeps` sweeps and returns the
new configurations with their log-amplitudes. `compute_expectation` returns
the local energies `E_L` and their mean `e_mean`, the energy estimate. The
optimizer turns `E_L` into a parameter update, which `apply_gradients` applies
with the learning rate `lr`.

The 30 steps take about ten seconds on a laptop CPU and show the energy
decreasing. Reaching the ground state, $E/N = -0.52862$ on this lattice
({doc}`guide/hamiltonian`), takes more steps and a larger network, as in
{doc}`resources/vit_wavefunction`.

## Fermions: the Hubbard model

The same five components, with occupation numbers in place of spins and an
ansatz built on a Slater determinant. The Hubbard model,

$$
H = -t \sum_{\langle i,j \rangle, \sigma} (c^\dagger_{i\sigma} c_{j\sigma} + \text{h.c.})
    + U \sum_i n_{i\uparrow} n_{i\downarrow} ,
$$

at half filling, one electron per site on average:

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

# 1. Lattice and Hamiltonian
L = 4
lattice = square(shape=(L, L))
N = lattice.Ns

t, U = 1.0, 4.0
H = hubbard_hamiltonian(lattice, nn=[((1, 0), t), ((0, 1), t)], U=U)

# 2. Initial configurations: N_mc chains, N/2 electrons of each spin
Ne = N
key = jax.random.key(0)
N_mc = 1024
key, subkey = jax.random.split(key)
occupations, N_up, N_down = init_config_spinful(subkey, Ns=N, Ne=Ne, N_mc=N_mc)
state = FermionState(occupations=occupations, Ne=Ne, lattice=lattice)

# 3. Wavefunction
model = FermionicTransformer(num_layers=1, d_model=16, num_heads=4, Ne=Ne, Ns=N)
key, subkey = jax.random.split(key)
params = model.init(subkey, state)
wf = WaveFunction(params=params, apply_fn=model.apply)

# 4. Monte Carlo move
action = BondExchange.create(lattice, Nbands=2)

# 5. Optimizer
optimizer = SR(diag_shift=1e-4, mode="complex")
opt_state = optimizer.init(wf.params)

N_steps, nsweeps, lr = 30, 1, 0.03
for step in range(N_steps):
    key, subkey = jax.random.split(key)
    mc_keys = jax.random.split(subkey, N_mc)

    state, log_amps, acceptance = sample(nsweeps, state, action, mc_keys, wf)
    E_L, e_mean, e2_mean = compute_expectation(H, wf, state, log_amps)

    updates, opt_state = optimizer(E_L, opt_state, state, wf)
    wf = wf.apply_gradients(updates, lr)

    print(f"step {step:3d}  E/N = {jnp.real(e_mean) / N: .4f}")
```

## Next steps

- {doc}`concepts`: the data types and functions used above.
- {doc}`guide/index`: each of the five components in depth, from custom
  lattices and Hamiltonians to other ansätze, moves and optimizer settings.
- {doc}`foundation_models`: one wavefunction trained on many Hamiltonians.
- {doc}`parallel`: the same scripts on several GPUs, on one node or many.
- {doc}`api`: the full reference.
