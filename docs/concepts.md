# Core Concepts

tachys is purely functional: values are immutable, and functions have no side
effects. Two kinds of object appear throughout — data, in the form of
immutable JAX pytrees, and pure functions mapping data to data. JAX's
transformations (`jit`, `vmap`, `grad`) presuppose this structure, so every
object in tachys can be passed to them directly.

This page describes the data type `State`, together with the `Lattice` it
carries, and the three functions that constitute a variational Monte Carlo
run: `sample`, `compute_expectation`, and an optimizer.

---

## The data: `State`

A `State` stores the configuration of the system — spins, or fermionic
occupation numbers — batched over the Monte Carlo chains run in parallel. It
holds two kinds of information:

- **Dynamic data**: the configurations, `state.spins` or
  `state.occupations`, of shape `(batch, ...)`. JAX traces, batches and
  differentiates these.
- **Static geometry**: `state.lattice`, a `Lattice` specifying the sites and
  their connectivity. It is stored as non-pytree metadata, shared by the whole
  batch and never differentiated.

`State` extends `flax.struct.PyTreeNode`, so any state is a valid pytree and
passes through `jit`, `vmap` and `grad` without wrapping.

```python
import jax.numpy as jnp
from tachys.lattice.lattice_database import chain
from tachys.lattice.spins.spin_state import SpinState
from tachys.lattice.fermions.fermion_state import FermionState

lattice = chain(4)                                    # a 4-site lattice
spins = jnp.array([[-1, 1, -1, 1]], dtype=jnp.int8)   # (batch=1, N=4)
s = SpinState(spins=spins, lattice=lattice)
s.Ns                                                   # 4 -- read from s.lattice, not a field

occ = jnp.array([[1, 0, 0, 1]], dtype=jnp.int8)        # (batch=1, 2*Ns)
f = FermionState(occupations=occ, Ne=2, lattice=chain(2))
```

`Ns` is a read-only property returning `state.lattice.Ns`. The constructor
takes the physical data (`spins` or `occupations`, together with `Ne` and
`Nbands` for fermions) and the `lattice`.

:::{important}
**States are never mutated.** A modified state is produced by `.replace(...)`,
which returns a new `State` and leaves the original unchanged. Every function
below follows this rule.
:::

---

## The functions: `sample`, `compute_expectation`, and the optimizer

A variational Monte Carlo run applies three pure functions in sequence at
every step.

1. **`sample`** advances the Markov chain. Given the current state, a
   wavefunction, a Monte Carlo move and a PRNG key, it returns a new state,
   the log-amplitudes evaluated on it, and the acceptance rate. The input
   state is unchanged.

   ```python
   state, log_amps, acceptance = sample(nsweeps, state, action, mc_keys, wf)
   ```

2. **`compute_expectation`** evaluates an operator. Given an operator, a
   wavefunction and a state, it computes the local estimator on each
   configuration and reduces it to global statistics. The state is read, not
   modified.

   ```python
   E_L, e_mean, e2_mean = compute_expectation(H, wf, state, log_amps)
   ```

3. **The optimizer** maps the local energies to a parameter update. `SR`,
   `SPRING` and `MARCH` are `flax.struct.PyTreeNode`s called as functions:

   ```python
   updates, opt_state = optimizer(E_L, opt_state, state, wf)
   wf = wf.apply_gradients(updates, lr)
   ```

   The optimizer returns the update and a new optimizer state;
   `apply_gradients` returns a new `WaveFunction` carrying the updated
   parameters.

Composed, the three are the entire training loop:

```python
for step in range(N_steps):
    key, subkey = jax.random.split(key)
    mc_keys = jax.random.split(subkey, N_mc)

    state, log_amps, acceptance = sample(1, state, action, mc_keys, wf)
    E_L, e_mean, e2_mean = compute_expectation(H, wf, state, log_amps)

    updates, opt_state = optimizer(E_L, opt_state, state, wf)
    wf = wf.apply_gradients(updates, lr)
```

Each iteration rebinds `state`, `wf` and `opt_state` to new values, as the
carry of a JAX `scan` would; no state is held anywhere else. A runnable
version is given in {doc}`quickstart`.

Training a single network across *many* Hamiltonians at once builds on
exactly this — see {doc}`foundation_models`.
