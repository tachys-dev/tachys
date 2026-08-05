# Foundation Models

Normally, one wavefunction is trained for one Hamiltonian. A **foundation
model** instead trains a single network across *many* Hamiltonians at
once — say, every value of a coupling constant you'd otherwise sweep one run
at a time — by mixing samples from all of them into one Monte Carlo batch.
The network reads off which system produced each sample and conditions its
output on that, so one trained model generalizes across the whole family
instead of one model per parameter value.

## The equations behind it

This is the setup introduced in Rende, Viteritti, Becca, Scardicchio, Laio &
Carleo, ["Foundation neural-network quantum states as a unified Ansatz for
multiple Hamiltonians"](https://www.nature.com/articles/s41467-025-62098-x),
*Nature Communications* (2025) — one of tachys's own authors. It's worth
reading alongside this page; here's the minimum needed to connect their
notation to tachys's objects.

The paper writes the wavefunction as $\psi_\theta(\sigma|\gamma)$: one network
with parameters $\theta$, taking both a configuration $\sigma$ and a coupling
vector $\gamma$ (e.g. $\gamma = J$ or $\gamma = U$) as input. That's exactly
`apply_fn(params, state)` here, once `state.system_couplings` supplies
$\gamma$ for every sample in the batch.

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

A tachys training step estimates exactly this by Monte Carlo: pick
$\mathcal{R}$ = `n_systems` values of $\gamma$ (a discrete stand-in for
$P(\gamma)$), sample $M/\mathcal{R}$ = `n_mc_per_system` configurations for
each, and average the local energy $E_L$ over the whole mixed batch of
$M$ = `N_mc` samples — which is exactly the `e_mean` that `compute_expectation`
returns.

Building the gradient correctly needs *per-system* averages rather than one
average over the whole mixed batch — e.g. the paper's per-system observable
average $\bar A_k = \frac{1}{M_k}\sum_j \langle\sigma_j|\hat A_{\gamma_k}|
\psi_\theta(\gamma_k)\rangle / \langle\sigma_j|\psi_\theta(\gamma_k)\rangle$.
That's `tachys.lattice.foundation.collectives.grouped_mean(x, state.system_ids,
state.n_systems)` — used internally by the SR-family optimizers whenever
`state` is a `FoundationState`, so centering happens system-by-system rather
than across the mixed batch.

## The data: two extra fields

This needs exactly two extra pieces of per-sample data, and no new machinery:

- `system_couplings`, shape `(N_mc, n_couplings)` — $\gamma$: the Hamiltonian
  parameters of the system each sample came from.
- `system_ids`, shape `(N_mc,)` — an integer in `[0, n_systems)` marking
  which system each sample belongs to.

`FoundationState` is a small `PyTreeNode` carrying exactly those two fields.
It's mixed into an ordinary `SpinState`/`FermionState` via multiple
inheritance to give `SpinFoundationState`/`FermionFoundationState` — still
just a `State`, so `sample`, `compute_expectation`, and the optimizer all work
on it completely unchanged (see {doc}`concepts` if those aren't familiar yet).

Building the batch happens once, before the loop: build one Hamiltonian per
system from the same template, `combine_systems` them into a single operator
whose couplings vary per sample, and `extract_system_couplings` reads that
same operator back out into the array `FoundationState.system_couplings`
expects.

## A full example: the Hubbard model across many `U`

One network, trained simultaneously on the Hubbard model at four values of the
on-site interaction — weakly correlated ($U=0$) all the way to strongly
correlated ($U=8$) — on the same 4×4 lattice at fixed filling.

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

This mirrors `tests/lattice/foundation/test_main_foundation_setup.py`, which
pins down exact expected values for each stage of this same pipeline (local
energies, sampled log-amplitudes, and optimizer updates) as a regression test
— worth a read if you want to see every intermediate value checked.

A couple of things worth noticing:

- `system_couplings` here has shape `(32, 1)`: only the on-site $U$ term
  varies across the four systems (the hopping amplitude $t$ is shared), so
  `extract_system_couplings` finds exactly one distinct varying column. That
  won't always be true — for a Hamiltonian where more than one *numeric*
  coupling value varies (like `heisenberg_hamiltonian`'s $J$ and $J/2$ terms),
  you'd see one column per distinct value, not one per physical parameter.
- The wavefunction just needs to read `lattice.system_couplings` alongside the
  configuration. `FermionFoundationRBM` (and `SpinFoundationRBM`) do exactly
  that, concatenating it onto the network's input before the ordinary
  backflow/RBM layer — everything else about them is unchanged from
  `FermionRBM`/`SpinRBM`.
- The training loop itself is untouched from {doc}`concepts`: `sample` and
  `compute_expectation` don't care that `state` is a `FoundationState` instead
  of a plain `FermionState`, and the optimizer already uses `system_ids`
  internally to center energies per-system rather than across the whole mixed
  batch.
