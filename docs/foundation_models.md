# Foundation Models

A **foundation model** is a single network trained on a family of
Hamiltonians at once, in place of one network per Hamiltonian. Samples from
every member of the family are mixed into one Monte Carlo batch, and the
network conditions its output on the couplings of the system each sample came
from, so one trained model covers the whole family — every value of a coupling
constant that would otherwise be swept one run at a time.

## Formulation

The construction follows Rende, Viteritti, Becca, Scardicchio, Laio & Carleo,
["Foundation neural-network quantum states as a unified Ansatz for multiple
Hamiltonians"](https://www.nature.com/articles/s41467-025-62098-x), *Nature
Communications* (2025).

The wavefunction is $\psi_\theta(\sigma|\gamma)$: one network with parameters
$\theta$, taking a configuration $\sigma$ and a coupling vector $\gamma$ (for
instance $\gamma = J$ or $\gamma = U$).

Training minimizes the energy averaged over an ensemble of Hamiltonians, drawn
from some distribution $P(\gamma)$ over coupling space:

$$
\mathcal{L}(\theta) = \int d\gamma\, P(\gamma)\,
\frac{\langle \psi_\theta(\gamma)|\hat H_\gamma|\psi_\theta(\gamma)\rangle}
     {\langle\psi_\theta(\gamma)|\psi_\theta(\gamma)\rangle}
= \int d\gamma\, P(\gamma) \sum_\sigma
\frac{|\psi_\theta(\sigma|\gamma)|^2}{\langle\psi_\theta(\gamma)|\psi_\theta(\gamma)\rangle}\,
E_L(\sigma,\gamma)
$$

A training step estimates this by Monte Carlo: $\mathcal{R}$ values
$\gamma_1, \dots, \gamma_\mathcal{R}$ discretize $P(\gamma)$,
$M_k = M/\mathcal{R}$ configurations are sampled from
$|\psi_\theta(\sigma|\gamma_k)|^2$ for each, and the local energy $E_L$ is
averaged over the mixed batch of $M$ samples.

The gradient requires *per-system* averages rather than a single average over
the mixed batch, for instance the per-system observable average
$\bar A_k = \frac{1}{M_k}\sum_{j \in k} \langle\sigma_j|\hat A_{\gamma_k}|\psi_\theta(\gamma_k)\rangle / \langle\sigma_j|\psi_\theta(\gamma_k)\rangle$,
the sum running over the samples of system $k$. Each Hamiltonian's energy is
normalized by its own $\langle\psi_\theta(\gamma_k)|\psi_\theta(\gamma_k)\rangle$,
so its gradient is a covariance over that system's samples alone. For real
parameters, with $O(\sigma,\gamma) = \nabla_\theta \log\psi_\theta(\sigma|\gamma)$,

$$
\nabla_\theta \mathcal{L} \approx \frac{1}{\mathcal{R}} \sum_{k=1}^{\mathcal{R}}
\frac{2}{M_k} \sum_{j \in k}
\mathrm{Re}\Big[\big(E_L(\sigma_j,\gamma_k) - \bar E_k\big)
\big(O(\sigma_j,\gamma_k) - \bar O_k\big)^*\Big]
$$

The local energies and the log-derivatives are therefore centered about the
mean of their own system, not about the mean of the mixed batch, and the
metric used by stochastic reconfiguration is centered the same way.

## The data: two extra fields

Two per-sample arrays are added:

- `system_couplings`, shape `(N_mc, n_couplings)` — $\gamma$, the Hamiltonian
  parameters of the system each sample came from.
- `system_ids`, shape `(N_mc,)` — an integer in `[0, n_systems)` identifying
  that system.

`FoundationState` is a `PyTreeNode` holding these two fields. Combined with
`SpinState` or `FermionState` by multiple inheritance, it gives
`SpinFoundationState` and `FermionFoundationState`; both remain `State`s, so
`sample`, `compute_expectation` and the optimizer apply to them unchanged (see
{doc}`concepts`).

The batch is built once, before the loop: one Hamiltonian per system from a
common template, `combine_systems` merges them into a single operator whose
couplings vary per sample, and `extract_system_couplings` returns that
operator's couplings as the array `FoundationState.system_couplings`
expects.

## Example: the Hubbard model at several values of `U`

One network trained on the Hubbard model at four values of the on-site
interaction, from the non-interacting limit ($U=0$) to the strongly correlated
regime ($U=8$), on a 4×4 lattice at fixed filling.

```python
import jax
import jax.numpy as jnp

from tachys.lattice.lattice_database import square
from tachys.lattice.fermions.fermion_state import init_config_spinful
from tachys.lattice.fermions.hamiltonians.hubbard import hubbard_square_pbc
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
Ne = 10                              # fixed filling, shared by every system

Us = [0.0, 2.0, 4.0, 8.0]            # gamma: one Hamiltonian per system, same template
n_systems = len(Us)
n_mc_per_system = 8
N_mc = n_systems * n_mc_per_system

Hs = [hubbard_square_pbc(L, U=U) for U in Us]
H = combine_systems(Hs, n_mc_per_system)
system_couplings = extract_system_couplings(H)
system_ids = jnp.repeat(jnp.arange(n_systems), n_mc_per_system)

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

N_steps, lr = 30, 5e-3
for step in range(N_steps):
    key, subkey = jax.random.split(key)
    mc_keys = jax.random.split(subkey, N_mc)

    state, log_amps, acceptance = sample(1, state, action, mc_keys, wf)
    E_L, e_mean, e2_mean = compute_expectation(H, wf, state, log_amps)

    updates, opt_state = optimizer(E_L, opt_state, state, wf)
    wf = wf.apply_gradients(updates, lr)

    print(f"step {step:3d}  E/N (ensemble avg) = {jnp.real(e_mean) / N: .4f}")
```

`tests/lattice/foundation/test_main_foundation_setup.py` runs this pipeline
as a regression test, with expected values for the local energies, the sampled
log-amplitudes and the optimizer updates.

Three properties of the example are worth stating explicitly.

**Couplings.** `system_couplings` has shape `(32, 1)`: only $U$ varies across
the four systems, the hopping $t$ being shared, so `extract_system_couplings`
finds one varying column. In general there is one column per distinct varying
*numeric* value, not one per physical parameter — `heisenberg_hamiltonian`
with $J$ and $J/2$ terms yields two.

**Ansatz.** The network reads `system_couplings` from the state alongside the
configuration. `FermionFoundationRBM` and `SpinFoundationRBM` concatenate it
onto the input of the backflow/RBM layer, and are otherwise identical to
`FermionRBM` and `SpinRBM`.

**Loop.** The training loop is unchanged from {doc}`concepts`: `sample` and
`compute_expectation` are indifferent to the state subtype, and the optimizer
uses `system_ids` to center energies per system.
