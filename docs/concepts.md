# Core Concepts

Tachys is built around three composable primitives: **states** that live in JAX's
pytree ecosystem, **operators** that are callable and composable via ordinary
arithmetic, and an **exact diagonalization engine** that ties them together.

---

## States

A `State` holds a *batch* of configurations. The first axis is the batch dimension;
the second indexes sites or modes. Extending `flax.struct.PyTreeNode` means every
state is a valid JAX pytree — it passes through `jit`, `vmap`, and `grad` without
any wrapping.

Tachys provides two concrete state types:

**`SpinState`** — for spin-½ systems. Spin values are integers in {−1, +1}.

```python
from tachys.lattice.spins.spin_state import SpinState
import jax.numpy as jnp

spins = jnp.array([[-1, 1, -1, 1]], dtype=jnp.int8)  # (batch=1, N=4)
s = SpinState(spins=spins, Ns=4)
```

**`FermionState`** — for spinful fermionic systems. Occupation numbers are binary
integers in {0, 1}. Modes are ordered as (site 0 ↑, site 0 ↓, site 1 ↑, …).

```python
from tachys.lattice.fermions.fermion_state import FermionState

occ = jnp.array([[1, 0, 0, 1]], dtype=jnp.int8)  # (batch=1, 2*Ns)
f = FermionState(occupations=occ, Ns=2, Ne=2)
```

:::{note}
Operators use `jax.vmap` internally. Pass a batch of 65 536 configurations and the
operator applies to all of them simultaneously — no Python loop required.
:::

---

## Operators as callables

An operator is an object with a `__call__` method. Call it on a batched state and
it returns one of two result types:

- **`DiagonalResult`** — for operators that do not connect different basis states
  (e.g. S^z, number operators). Carries only matrix elements.
- **`OffdiagonalResult`** — for operators that reach a new basis state
  (e.g. S^+, hopping terms). Carries the connected state, a validity mask, and
  matrix elements.

```python
from tachys.lattice.spins.spin_operators import Sz, Splus

sz0 = Sz(site=0)    # diagonal
sp1 = Splus(site=1) # off-diagonal

diag    = sz0(state)   # DiagonalResult
offdiag = sp1(state)   # OffdiagonalResult

# DiagonalResult fields
diag.matrix_element          # shape (batch,)

# OffdiagonalResult fields
offdiag.connected_states     # shape (batch, N)  — the reached configuration
offdiag.mask                 # shape (batch,)    — False when operator annihilates
offdiag.matrix_element       # shape (batch,)
```

The base class `_Operator` handles the `vmap` call. Subclasses only implement
`apply(state)`, which operates on a single configuration (no batch dimension).

---

## Operator algebra

Operators support arithmetic. The resulting composite is itself a callable —
there is no distinction between a primitive operator and a Hamiltonian built from
hundreds of terms.

| Expression | Result type | Meaning |
|------------|-------------|---------|
| `A + B` | `_OperatorSum` | Sum of two operators |
| `s * A` or `A * s` | `_Operator` | Rescale coupling by scalar `s` |
| `A * B` | `_OperatorMul` | Sequential product |

```python
from tachys.lattice.spins.spin_operators import Sz, Splus, Sminus
from tachys.lattice.spins.hamiltonians.heisenberg import heisenberg_square_pbc

# Scalar multiplication
half_sz = 0.5 * Sz(site=0)

# Operator sum — build any Hamiltonian term by term
ising = Sz(site=0) * Sz(site=1) + Sz(site=1) * Sz(site=2)

# Operator product — sequential application with correct fermionic signs
hopping = Splus(site=0) * Sminus(site=1)

# Compose with any existing Hamiltonian
H = heisenberg_square_pbc(L=4)
H_field = H + 0.1 * Sz(site=0)  # add a staggered field on site 0
result  = H_field(state)         # DiagOffdiagResult — same interface as always
```

:::{tip}
Because operators are `PyTreeNode`s, their coupling constants are leaves in the
pytree. You can differentiate through them with `jax.grad`, which enables
variational optimization over Hamiltonian parameters.
:::

---

## Built-in Hamiltonians

Tachys ships two Hamiltonians on the square lattice with periodic boundary
conditions. Each returns an operator that can be composed, scaled, or perturbed.

**Heisenberg model**

$$
H = J \sum_{\langle i,j \rangle} \left[ S^z_i S^z_j + \tfrac{1}{2}(S^+_i S^-_j + S^-_i S^+_j) \right]
$$

```python
from tachys.lattice.spins.hamiltonians.heisenberg import heisenberg_square_pbc

H = heisenberg_square_pbc(L=4, J=1.0)
```

**Hubbard model**

$$
H = -t \sum_{\langle i,j \rangle, \sigma} (c^\dagger_{i\sigma} c_{j\sigma} + \text{h.c.})
    + U \sum_i n_{i\uparrow} n_{i\downarrow}
$$

```python
from tachys.lattice.fermions.hamiltonians.hubbard import hubbard_square_pbc

H = hubbard_square_pbc(L=4, t=1.0, U=8.0)
```

Sites are indexed row-major in both cases: site at (x, y) maps to `x·L + y`.

---

## Exact diagonalization

`exact_diag` constructs the sparse Hamiltonian matrix from all diagonal and
off-diagonal contributions and computes the `k` lowest eigenvalues via
`scipy.sparse.linalg.eigsh`.

The only thing you provide beyond the operator and the basis is a **pack** function:
a map from a batch of states to unique integer indices. This is the sole coupling
between your state representation and the sparse matrix structure.

```python
from tachys.lattice.exact_diag import exact_diag, spins_hilbert_space
import numpy as np

all_configs = spins_hilbert_space(N=16)          # shape (65536, 16)
state_full  = SpinState(spins=jnp.array(all_configs), Ns=16)

def pack(state):
    bits = (np.asarray(state.spins) + 1) // 2   # {-1,+1} → {0,1}
    return (bits * 2 ** np.arange(bits.shape[-1])).sum(axis=-1)

eigenvalues, eigenvectors = exact_diag(state_full, H, pack, k=1)
# eigenvalues : shape (1,) — ascending
# eigenvectors: shape (65536, 1)
```

The `pack` function must be injective over the Hilbert space — each configuration
must map to a distinct non-negative integer. Beyond that, the choice is yours.
