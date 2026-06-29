# Examples

## Heisenberg model on a 4×4 lattice

Compute the ground state energy of the antiferromagnetic Heisenberg model and
compare to the known value E₀/N ≈ −0.7018 (Sandvik, 1997).

```python
import numpy as np
import jax.numpy as jnp
from tachys.lattice.spins.spin_state import SpinState
from tachys.lattice.spins.hamiltonians.heisenberg import heisenberg_square_pbc
from tachys.lattice.exact_diag import spins_hilbert_space, exact_diag

L, N = 4, 16
H = heisenberg_square_pbc(L=L, J=1.0)

all_configs = spins_hilbert_space(N)
state_full  = SpinState(spins=jnp.array(all_configs, dtype=jnp.int8), Ns=N)

def pack(state):
    bits = (np.asarray(state.spins) + 1) // 2
    return (bits * 2 ** np.arange(bits.shape[-1])).sum(axis=-1)

eigenvalues, _ = exact_diag(state_full, H, pack, k=1)

print(f"E₀     = {eigenvalues[0]:.8f}")      # -11.22845248
print(f"E₀ / N = {eigenvalues[0] / N:.8f}")  # -0.70177828
```

---

## Hubbard model at quarter filling

4×4 lattice, N_e = 4 electrons, benchmarked against Dagotto et al. (1992) at
U/t = 4 and U/t = 8.

```python
import numpy as np
import jax.numpy as jnp
from tachys.lattice.fermions.fermion_state import FermionState
from tachys.lattice.fermions.hamiltonians.hubbard import hubbard_square_pbc
from tachys.lattice.exact_diag import fermions_hilbert_space, exact_diag

L, Ns, Ne = 4, 16, 4

all_configs = fermions_hilbert_space(Ns, Ne)   # C(32, 4) = 35 960 states
state_full  = FermionState(
    occupations=jnp.array(all_configs, dtype=jnp.int8),
    Ns=Ns,
    Ne=Ne,
)

def pack(state):
    occ = np.asarray(state.occupations)
    return (occ * 2 ** np.arange(occ.shape[-1])).sum(axis=-1)

for U in (4.0, 8.0):
    H = hubbard_square_pbc(L=L, t=1.0, U=U)
    eigenvalues, _ = exact_diag(state_full, H, pack, k=1)
    print(f"U={U:.0f}:  E₀ = {eigenvalues[0]:.5f}")

# U=4:  E₀ = -11.53029    (Dagotto 1992: -11.5303)
# U=8:  E₀ = -11.32150    (Dagotto 1992: -11.3215)
```

---

## Adding a perturbation

Because operators compose via ordinary arithmetic, adding a perturbation to any
existing Hamiltonian is a single line:

```python
from tachys.lattice.spins.spin_operators import Sz
from tachys.lattice.spins.hamiltonians.heisenberg import heisenberg_square_pbc

H       = heisenberg_square_pbc(L=4, J=1.0)
H_field = H + 0.1 * Sz(site=0)   # uniform field on site 0

# H_field is itself a callable — use it exactly like H
eigenvalues, _ = exact_diag(state_full, H_field, pack, k=1)
```
