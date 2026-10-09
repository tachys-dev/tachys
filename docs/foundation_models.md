# Foundation Models

A foundation model, or foundation neural-network quantum state (Foundation
NQS), is one network trained on a family of Hamiltonians at once, instead of
one network per Hamiltonian. The network takes the couplings of the
Hamiltonian as an input, next to the configuration, so a single optimization
covers the whole family: the Hubbard model at several values of $U$, for
instance, or the parameter space relevant to a given material.

## Formulation

The method is that of Rende, Viteritti, Becca, Scardicchio, Laio & Carleo,
["Foundation neural-network quantum states as a unified Ansatz for multiple
Hamiltonians"](https://www.nature.com/articles/s41467-025-62098-x), *Nature
Communications* (2025).

The wavefunction $\psi_\theta(\sigma|\gamma)$ depends on the configuration
$\sigma$ and on the couplings $\gamma$ of the Hamiltonian $\hat H_\gamma$. The
parameters $\theta$ minimize the energy averaged over a distribution
$P(\gamma)$ of couplings,

$$
\mathcal{L}(\theta) = \int d\gamma\, P(\gamma)\,
\frac{\langle \psi_\theta(\gamma)|\hat H_\gamma|\psi_\theta(\gamma)\rangle}
     {\langle\psi_\theta(\gamma)|\psi_\theta(\gamma)\rangle} .
$$

In practice, $P(\gamma)$ is replaced by $\mathcal{R}$ values
$\gamma_1, \dots, \gamma_\mathcal{R}$, and each step samples
$M_k = M/\mathcal{R}$ configurations from $|\psi_\theta(\sigma|\gamma_k)|^2$
for each system $k$. Each energy has its own normalization
$\langle\psi_\theta(\gamma_k)|\psi_\theta(\gamma_k)\rangle$, so the gradient is
a sum of covariances, one per system. With
$O(\sigma,\gamma) = \nabla_\theta \log\psi_\theta(\sigma|\gamma)$ and real
parameters,

$$
\nabla_\theta \mathcal{L} \approx \frac{1}{\mathcal{R}} \sum_{k=1}^{\mathcal{R}}
\frac{2}{M_k} \sum_{j \in k}
\mathrm{Re}\Big[\big(E_L(\sigma_j,\gamma_k) - \bar E_k\big)
\big(O(\sigma_j,\gamma_k) - \bar O_k\big)^*\Big] ,
$$

where $j$ runs over the samples of system $k$, and $\bar E_k$ and $\bar O_k$
are averages over these samples. Compared with a single Hamiltonian, the only
change is that local energies and log-derivatives are centred on the mean of
their own system, not of the whole batch. The optimizers do this automatically,
for the local energies and for the kernel of stochastic reconfiguration.

## What changes in a script

`sample`, `compute_expectation`, the optimizers and the training loop work
unchanged. Three objects change:

- **The state** records the system of each sample. `SpinFoundationState` and
  `FermionFoundationState` are a `SpinState` and a `FermionState` with three
  more fields: `system_ids`, of shape `(N_mc,)`, the index $k$ of the system of
  each sample; `system_couplings`, of shape `(N_mc, n_couplings)`, its
  couplings $\gamma_k$; and `n_systems`, the number of systems $\mathcal{R}$.
- **The Hamiltonian** is a single operator whose couplings change from sample
  to sample. `combine_systems(Hs, n_mc_per_system)` builds it from a list `Hs`
  of Hamiltonians made by the same factory, one per system: the first
  `n_mc_per_system` samples see `Hs[0]`, the next `n_mc_per_system` see
  `Hs[1]`, and so on. `extract_system_couplings(H)` returns the couplings that
  differ between the systems, as the array `system_couplings`.
- **The ansatz** reads `state.system_couplings` along with the configuration.
  `SpinFoundationRBM` and `FermionFoundationRBM` are `SpinRBM` and
  `FermionRBM` with the couplings appended to the input of their first dense
  layer.

## Example: the Hubbard model at four values of $U$

One network for the Hubbard model at $U = 0, 2, 4, 8$, on the $4 \times 4$
lattice with 10 electrons, and 256 samples per system:

```python
import jax
import jax.numpy as jnp

from tachys.lattice.lattice_database import square
from tachys.lattice.fermions.fermion_state import init_config_spinful
from tachys.lattice.fermions.hamiltonians.hubbard import hubbard_hamiltonian
from tachys.lattice.foundation.foundation_state import FermionFoundationState
from tachys.lattice.foundation.operators import combine_systems, extract_system_couplings
from tachys.lattice.ansatz.rbm_foundation import FermionFoundationRBM
from tachys.lattice.bond_exchange import BondExchange
from tachys.wavefunction import WaveFunction
from tachys.optimizer import SR
from tachys.montecarlo import sample
from tachys.lattice.operator.local_estimator import compute_expectation

L = 4
lattice = square(shape=(L, L))
N = lattice.Ns
Ne = 10                                 # the same for every system

Us = [0.0, 2.0, 4.0, 8.0]
n_systems = len(Us)
n_mc_per_system = 256
N_mc = n_systems * n_mc_per_system

# one Hamiltonian per system, from the same factory
Hs = [hubbard_hamiltonian(lattice, nn=[((1, 0), 1.0), ((0, 1), 1.0)], U=U) for U in Us]
H = combine_systems(Hs, n_mc_per_system)
system_couplings = extract_system_couplings(H)                   # (N_mc, 1): U
system_ids = jnp.repeat(jnp.arange(n_systems), n_mc_per_system)  # (N_mc,): 0, ..., 0, 1, ...

key = jax.random.key(0)
key, subkey = jax.random.split(key)
occupations, N_up, N_down = init_config_spinful(subkey, Ns=N, Ne=Ne, N_mc=N_mc)
state = FermionFoundationState(
    occupations=occupations, Ne=Ne, lattice=lattice,
    system_couplings=system_couplings, system_ids=system_ids, n_systems=n_systems,
)

model = FermionFoundationRBM(hidden_units=64)
key, subkey = jax.random.split(key)
params = model.init(subkey, state)
wf = WaveFunction(params=params, apply_fn=model.apply)

action = BondExchange.create(lattice, Nbands=2)
optimizer = SR(diag_shift=1e-3, mode="real")
opt_state = optimizer.init(wf.params)

N_steps, lr = 100, 0.05
for step in range(N_steps):
    key, subkey = jax.random.split(key)
    mc_keys = jax.random.split(subkey, N_mc)

    state, log_amps, acceptance = sample(1, state, action, mc_keys, wf)
    E_L, e_mean, e2_mean = compute_expectation(H, wf, state, log_amps)

    updates, opt_state = optimizer(E_L, opt_state, state, wf)
    wf = wf.apply_gradients(updates, lr)

    if step % 20 == 0:
        E_per_system = jnp.real(E_L).reshape(n_systems, n_mc_per_system).mean(axis=1)
        print(f"step {step:3d}  E/N = {(E_per_system / N).round(4)}")
```

```text
step   0  E/N = [0.0207 0.2234 0.4364 0.8336]
step  20  E/N = [-1.1259 -1.1267 -1.0346 -0.8497]
step  40  E/N = [-1.4763 -1.319  -1.1918 -0.9727]
step  60  E/N = [-1.495  -1.3339 -1.2153 -1.0136]
step  80  E/N = [-1.4933 -1.3289 -1.2057 -1.0356]
```

Each line gives the energy per site of the four systems, in the order of `Us`.
The samples of each system are consecutive, so reshaping `E_L` separates them;
`e_mean`, the mean over the whole batch, mixes the four energies. At $U = 0$
the energy approaches the exact value for free fermions, $E/N = -1.5$. The
100 steps take about 20 seconds on the CPU of a laptop (Apple M4 Max).

`system_couplings` has a single column, holding $U$:
`extract_system_couplings` keeps only the couplings that differ between the
systems, and the hopping is the same in all four. A parameter that appears with
two different prefactors gives two columns: for a family of
`heisenberg_hamiltonian` models at different $J$, they hold $J$, the
coefficient of $S^zS^z$, and $J/2$, that of the exchange term.
