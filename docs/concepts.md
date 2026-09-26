# Core Concepts

tachys keeps data and functions apart. The data (configurations, parameters,
optimizer state) are immutable JAX pytrees. The functions that run a
simulation take data and return new data, without modifying their arguments;
they are compiled with `jit` and run in parallel on all available devices.

## The data

Three pytrees hold everything that changes during a run:

- a `State`, the configurations of the Markov chains;
- a `WaveFunction`, the parameters of the network together with its `apply`
  function ({doc}`guide/wavefunctions`);
- the optimizer state, whatever the optimizer carries from one step to the
  next ({doc}`guide/optimization`).

A `State` holds the configurations of `N_mc` chains, spins or occupation
numbers, in an array whose leading axis runs over the chains. It also carries
the `Lattice` ({doc}`guide/lattices`) as static metadata: one lattice for the
whole batch, which JAX never traces or differentiates.

```python
import jax.numpy as jnp
from tachys.lattice.lattice_database import chain
from tachys.lattice.spins.spin_state import SpinState
from tachys.lattice.fermions.fermion_state import FermionState

# two chains of 4 spins
spins = jnp.array([[1, -1, 1, -1],
                   [1, 1, -1, -1]], dtype=jnp.int8)       # (N_mc, Ns)
s = SpinState(spins=spins, lattice=chain(4))
s.Ns                                                       # 4, read from the lattice

# one chain of 2 electrons on 2 sites: ↑ on site 0, ↓ on site 1
occupations = jnp.array([[1, 0, 0, 1]], dtype=jnp.int8)   # (N_mc, 2 Ns)
f = FermionState(occupations=occupations, Ne=2, lattice=chain(2))
```

A state is never modified in place: `replace` returns a new state, and leaves
the original unchanged.

```python
flipped = s.replace(spins=-s.spins)     # s is unchanged
```

## The functions

A step of variational Monte Carlo applies three functions in turn.

1. **`sample`** advances the Markov chains by `nsweeps` sweeps of the move
   `action`, with one PRNG key per chain. It returns the new configurations,
   their log-amplitudes and the acceptance rate:

   ```python
   state, log_amps, acceptance = sample(nsweeps, state, action, mc_keys, wf)
   ```

2. **`compute_expectation`** evaluates an operator on the sample. It returns
   the local estimator on every configuration, its mean over the sample (the
   estimate of the expectation value), and the mean of its squared modulus:

   ```python
   E_L, e_mean, e2_mean = compute_expectation(H, wf, state, log_amps)
   ```

3. **The optimizer** turns the local energies into a parameter update and a new
   optimizer state. `apply_gradients` then returns a new `WaveFunction` with
   the updated parameters:

   ```python
   updates, opt_state = optimizer(E_L, opt_state, state, wf)
   wf = wf.apply_gradients(updates, lr)
   ```

Together they make the training loop:

```python
for step in range(N_steps):
    key, subkey = jax.random.split(key)
    mc_keys = jax.random.split(subkey, N_mc)

    state, log_amps, acceptance = sample(1, state, action, mc_keys, wf)
    E_L, e_mean, e2_mean = compute_expectation(H, wf, state, log_amps)

    updates, opt_state = optimizer(E_L, opt_state, state, wf)
    wf = wf.apply_gradients(updates, lr)
```

Each iteration rebinds `state`, `wf` and `opt_state` to new values, like the
carry of a `jax.lax.scan`; nothing is stored anywhere else. The
{doc}`quickstart` gives runnable versions, and {doc}`foundation_models` trains
one network on many Hamiltonians with the same loop.
