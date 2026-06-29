# Quickstart

This page gets you from install to a ground-state energy in under five minutes.
The example uses the antiferromagnetic Heisenberg model on a 4×4 square lattice —
the same system used as a benchmark throughout the library.

## Three steps

Every Tachys workflow follows the same pattern:

1. **Build an operator.** Call a Hamiltonian factory; the result is a callable.
2. **Generate the basis.** Enumerate all valid configurations in the Hilbert space.
3. **Diagonalize.** Pass the basis, the operator, and an index function to `exact_diag`.

## Full example

```python
import numpy as np
import jax.numpy as jnp
from tachys.lattice.spins.spin_state import SpinState
from tachys.lattice.spins.hamiltonians.heisenberg import heisenberg_square_pbc
from tachys.lattice.exact_diag import spins_hilbert_space, exact_diag

L, N = 4, 16

# Step 1 — build the Hamiltonian (an operator is just a callable)
H = heisenberg_square_pbc(L=L, J=1.0)

# Step 2 — generate the complete 2^16-dimensional basis
all_configs = spins_hilbert_space(N)                      # shape (65536, 16)
state_full  = SpinState(spins=jnp.array(all_configs, dtype=jnp.int8), Ns=N)

# Step 3 — define an injective index function and diagonalize
def pack(state):
    bits = (np.asarray(state.spins) + 1) // 2            # {-1,+1} → {0,1}
    return (bits * 2 ** np.arange(bits.shape[-1])).sum(axis=-1)

eigenvalues, eigenvectors = exact_diag(state_full, H, pack, k=1)

print(f"E₀     = {eigenvalues[0]:.6f}")     # -11.228452
print(f"E₀ / N = {eigenvalues[0] / N:.6f}") # -0.701778
```

## What just happened?

- `heisenberg_square_pbc` returns an `_OperatorSum` — a callable that, when called
  on a batch of states, returns diagonal and off-diagonal matrix elements for every
  state in the batch simultaneously (via `jax.vmap`).
- `spins_hilbert_space` generates all 2^N spin configurations as a NumPy array.
- `exact_diag` applies `H` to every row of that array, assembles a sparse COO
  matrix, and calls `scipy.sparse.linalg.eigsh` for the lowest `k` eigenvalues.
- The `pack` function is the only coupling between your state representation and
  the matrix indices. It must be injective — each configuration maps to a unique
  non-negative integer.

## Next steps

- {doc}`concepts` — understand states, operators, and the algebra in depth.
- {doc}`api` — full API reference.
- {doc}`examples` — worked examples for Heisenberg and Hubbard models.
