# Core Concepts

Tachys is written in a **purely functional** style. There are no objects that
quietly hold mutable state, and nothing you call ever changes what you pass
into it. Everything you touch is one of two things:

- **Data** — an immutable value, almost always a JAX pytree.
- **A pure function** — something that takes data in and returns new data
  out, without side effects.

This isn't an implementation detail you can ignore: JAX's own transformations
(`jit`, `vmap`, `grad`) already require this discipline, so tachys leans into
it instead of working around it. Once it clicks, the whole library reads the
same way: a handful of data types, and a small set of functions that turn one
piece of data into another.

This page introduces both halves — the data (`State`, and the `Lattice` it
carries) and the functions that manipulate it, in particular the three you'll
use in almost every script: `sample`, `compute_expectation`, and an optimizer.

---

## The data: `State`

A `State` is the physical configuration of your system — spins, or fermionic
occupation numbers — batched over however many independent Monte Carlo chains
you're running at once. It bundles two very different kinds of information:

- **Dynamic data**: the actual configurations (`state.spins` or
  `state.occupations`), shape `(batch, ...)`. This is what JAX traces,
  batches, and differentiates through.
- **Static geometry**: `state.lattice`, a `Lattice` describing where the sites
  are and how they're connected. It's attached to every state but marked as
  non-pytree metadata — shared by the whole batch, not itself something you
  differentiate through.

Because `State` extends `flax.struct.PyTreeNode`, every state is automatically
a valid JAX pytree: it passes through `jit`, `vmap`, and `grad` with no
wrapping or special-casing.

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

Note that `Ns` is not something you pass in: it's a read-only property
computed as `state.lattice.Ns`. A `State`'s only real constructor arguments
are its physical data (`spins`/`occupations`, plus `Ne`/`Nbands` for fermions)
and the `lattice` it lives on.

:::{important}
**States are never mutated in place.** Anything that "changes" a state —
inside tachys or in your own code — does so by calling `.replace(...)`, which
returns a brand-new `State` and leaves the original completely untouched. You
will see this pattern everywhere, starting with the functions below.
:::

---

## The functions: `sample`, `compute_expectation`, and the optimizer

A VMC run is nothing more than three pure functions, called in a loop, each
one handing its output to the next:

1. **`sample`** advances the Markov chain. Given the current `state`, a
   wavefunction, a Monte Carlo move, and a PRNG key, it returns a **new**
   state — the input `state` is left untouched.

   ```python
   state, log_amps, acceptance = sample(nsweeps, state, action, mc_keys, wf)
   ```

2. **`compute_expectation`** turns a state into a number. Given an operator, a
   wavefunction, and a state, it evaluates the operator's local estimator and
   reduces it to global statistics. It only *reads* `state` — there's no new
   state coming out the other end, only energies.

   ```python
   E_L, e_mean, e2_mean = compute_expectation(H, wf, state, log_amps)
   ```

3. **The optimizer** turns those energies into a parameter update. Optimizers
   like `SR`, `SPRING`, and `MARCH` are themselves just data — plain
   `flax.struct.PyTreeNode`s — called as functions:

   ```python
   updates, opt_state = optimizer(E_L, opt_state, state, wf)
   wf = wf.apply_gradients(updates, lr)
   ```

   `optimizer(...)` doesn't touch `wf` or `opt_state` — it returns new values
   for both. `wf.apply_gradients` is the same story: it returns a new
   `WaveFunction` with updated parameters, rather than editing the one you
   already have.

Put the three together and you have the entire training loop — nothing else
is hidden underneath:

```python
for step in range(N_steps):
    key, subkey = jax.random.split(key)
    mc_keys = jax.random.split(subkey, N_mc)

    state, log_amps, acceptance = sample(1, state, action, mc_keys, wf)
    E_L, e_mean, e2_mean = compute_expectation(H, wf, state, log_amps)

    updates, opt_state = optimizer(E_L, opt_state, state, wf)
    wf = wf.apply_gradients(updates, lr)
```

Every variable on the left-hand side is a *new* value each iteration —
`state`, `wf`, and `opt_state` are simply reassigned, exactly like the carry
of a JAX `scan`, just written out as an ordinary Python loop. There's no
`Trainer` object accumulating hidden state behind the scenes; if you want to
know what a tachys training run does, this loop *is* the answer. See
{doc}`quickstart` for a complete, runnable version of it.

Training a single network across *many* Hamiltonians at once builds on
exactly this — see {doc}`foundation_models`.
