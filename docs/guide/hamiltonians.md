# Hamiltonians

A Hamiltonian in tachys is an operator: applied to a batch of configurations
$x$, it returns the configurations $x'$ that it connects to each of them,
together with the matrix elements between them. That is all the local energy
requires,

$$
E_L(x) = \sum_{x'} \langle x|H|x'\rangle\, \frac{\psi(x')}{\psi(x)} ,
$$

and `compute_expectation` evaluates it for any operator: a Hamiltonian, or any
other observable.

Hamiltonians are sums of products of elementary operators, listed
{ref}`below <operators-in-tachys>`, which combine like the symbols of a
formula:

- `c * A`, with `c` a Python number, multiplies `A` by `c`;
- `A * B` is the product $AB$, with the factors in the order of the formula;
- `A + B` is the sum.

There is no subtraction or negation: `A - h * B` is written `A + (-h) * B`.
Products of sums are expanded by hand: `(A + B) * C` is written
`A * C + B * C`.

## Spin models

### A single bond

The exchange interaction between the spins on sites 0 and 1,

$$
\mathbf{S}_0\cdot\mathbf{S}_1 = S^z_0S^z_1 + \tfrac12\,\big(S^+_0S^-_1 + S^-_0S^+_1\big),
$$

translates symbol by symbol:

```python
from tachys.lattice.spins.spin_operators import Sz, Splus, Sminus

bond = Sz(0) * Sz(1) + 0.5 * Splus(0) * Sminus(1) + 0.5 * Sminus(0) * Splus(1)
```

`bond` is already an operator: `compute_expectation(bond, wf, state, log_amps)`
returns $\langle \mathbf{S}_0\cdot\mathbf{S}_1 \rangle$ on the current sample.

### The Heisenberg model

On a lattice, the Hamiltonian repeats this term on every bond. The
$J_1$-$J_2$ model of the {doc}`../quickstart`,

$$
H = \sum_{\langle i,j\rangle} J_{ij} \Big[ S^z_iS^z_j + \tfrac12\,\big(S^+_iS^-_j + S^-_iS^+_j\big) \Big],
$$

has $J_{ij} = J_1$ between nearest neighbours and $J_{ij} = J_2$ across the
diagonals. On an $L \times L$ square lattice, the site in column $x$ and row
$y$ has index $x + L\,y$: this is the numbering of `square(shape=(L, L))`
({doc}`lattices`), used by the states of the quickstart. With periodic
boundaries, the coordinates are taken modulo $L$. Pairing every site with its
neighbours to the right and above, and with its two diagonal neighbours on the
right, counts each bond once:

```python
from tachys.lattice.spins.spin_operators import Sz, Splus, Sminus

L = 4                                   # L × L sites, periodic boundaries
J1, J2 = 1.0, 0.5

H = None
for x in range(L):
    for y in range(L):
        i          = x + L * y                          # site (x, y)
        right      = (x + 1) % L + L * y                # site (x + 1, y)
        up         = x + L * ((y + 1) % L)              # site (x, y + 1)
        up_right   = (x + 1) % L + L * ((y + 1) % L)    # site (x + 1, y + 1)
        down_right = (x + 1) % L + L * ((y - 1) % L)    # site (x + 1, y - 1)
        for j, J in [(right, J1), (up, J1), (up_right, J2), (down_right, J2)]:
            term = (J * Sz(i) * Sz(j)
                    + (J / 2) * Splus(i) * Sminus(j)
                    + (J / 2) * Sminus(i) * Splus(j))
            H = term if H is None else H + term
```

The sum starts from `None` because an operator cannot be added to the number 0.
Terms of the same form are merged as they are added: `H` holds three vectorized
terms, one each for $S^zS^z$, $S^+S^-$ and $S^-S^+$ on all bonds, however large
the lattice. Writing a Hamiltonian term by term therefore costs nothing at run
time.

On a cluster this small, exact diagonalization checks the construction. The
basis holds all $2^{16}$ configurations of the 16 spins, and `pack` labels each
of them with a distinct integer:

```python
import numpy as np
import jax.numpy as jnp
from tachys.lattice.lattice_database import square
from tachys.lattice.exact_diag import exact_diag, spins_hilbert_space
from tachys.lattice.spins.spin_state import SpinState

N = 16                                  # 4 × 4 spins
all_spins = jnp.asarray(spins_hilbert_space(N), dtype=jnp.int8)   # (2**N, N)
basis = SpinState(spins=all_spins, lattice=square(shape=(4, 4)))

def pack(state):
    bits = (np.asarray(state.spins) + 1) // 2
    return bits @ 2 ** np.arange(bits.shape[-1])

E0, _ = exact_diag(basis, H, pack)
E0[0] / N                 # -0.5286202, the exact energy per site
```

`exact_diag` needs a Hamiltonian with both diagonal and off-diagonal terms.

### The transverse-field Ising model

$$
H = -J \sum_{\langle i,j\rangle} \sigma^z_i\sigma^z_j - h \sum_i \sigma^x_i
$$

is written with Pauli matrices, $\sigma^\alpha = 2S^\alpha$, and its field is a
single-site term, added once for each site:

```python
from tachys.lattice.spins.spin_operators import Sz, Sx

L = 4                                   # L × L sites, periodic boundaries
J, h = 1.0, 2.0

H = None
for x in range(L):
    for y in range(L):
        i     = x + L * y               # site (x, y)
        right = (x + 1) % L + L * y     # site (x + 1, y)
        up    = x + L * ((y + 1) % L)   # site (x, y + 1)
        for j in [right, up]:
            term = (-4 * J) * Sz(i) * Sz(j)         # -J σ^z_i σ^z_j
            H = term if H is None else H + term
        H = H + (-2 * h) * Sx(i)                    # -h σ^x_i
```

`Sx` flips one spin at a time, so this model does not conserve $S^z$; it is
sampled with `SpinFlip`, see {doc}`sampling`.

## Fermionic models

### The Hubbard model

$$
H = -t \sum_{\langle i,j\rangle,\sigma} \big(c^\dagger_{i\sigma}c_{j\sigma} + c^\dagger_{j\sigma}c_{i\sigma}\big) + U \sum_i n_{i\uparrow}n_{i\downarrow}
$$

```python
from tachys.lattice.fermions.fermion_operators import (
    Cup, Cup_dag, Cdn, Cdn_dag, Nup, Ndn,
)

L = 4                                   # L × L sites, periodic boundaries
t, U = 1.0, 8.0

H = None
for x in range(L):
    for y in range(L):
        i     = x + L * y               # site (x, y)
        right = (x + 1) % L + L * y     # site (x + 1, y)
        up    = x + L * ((y + 1) % L)   # site (x, y + 1)
        for j in [right, up]:
            hop_up = (-t) * Cup_dag(i) * Cup(j) + (-t) * Cup_dag(j) * Cup(i)
            hop_dn = (-t) * Cdn_dag(i) * Cdn(j) + (-t) * Cdn_dag(j) * Cdn(i)
            term = hop_up + hop_dn
            H = term if H is None else H + term
        H = H + U * Nup(i) * Ndn(i)
```

`Cup_dag(i) * Cup(j)` is $c^\dagger_{i\uparrow}c_{j\uparrow}$, in the same
order. The fermionic operators include the Jordan–Wigner sign, with the modes
ordered as in {doc}`configurations`.

A repulsion $V$ between electrons on neighbouring sites,
$V\sum_{\langle i,j\rangle} n_i n_j$ with $n_i = n_{i\uparrow} + n_{i\downarrow}$,
is a product of two sums, so it is expanded into four products, added bond by
bond to the Hubbard Hamiltonian `H` above:

```python
from tachys.lattice.fermions.fermion_operators import Nup, Ndn

L = 4                                   # L × L sites, periodic boundaries
V = 1.0

for x in range(L):
    for y in range(L):
        i     = x + L * y               # site (x, y)
        right = (x + 1) % L + L * y     # site (x + 1, y)
        up    = x + L * ((y + 1) % L)   # site (x, y + 1)
        for j in [right, up]:
            H = H + (V * Nup(i) * Nup(j) + V * Nup(i) * Ndn(j)
                     + V * Ndn(i) * Nup(j) + V * Ndn(i) * Ndn(j))
```

## Built-in models

The three models above are also provided as factories, on any lattice and with
any list of bonds. They take a `Lattice` ({doc}`lattices`) in place of the
explicit indices, and build the same operators with index arrays:

| Factory | Hamiltonian |
|---|---|
| `heisenberg_hamiltonian(lat, nn)` | $\sum_{\langle i,j\rangle} J_{ij}\, \mathbf{S}_i\cdot\mathbf{S}_j$ |
| `ising_transverse_field_hamiltonian(lat, nn, h)` | $-\sum_{\langle i,j\rangle} J_{ij}\,\sigma^z_i\sigma^z_j - h\sum_i \sigma^x_i$ |
| `hubbard_hamiltonian(lat, nn, U)` | $-\sum_{\langle i,j\rangle,\sigma} t_{ij}\,(c^\dagger_{i\sigma}c_{j\sigma} + \text{h.c.}) + U\sum_i n_{i\uparrow}n_{i\downarrow}$ |

They live in `tachys.lattice.spins.hamiltonians.heisenberg`,
`tachys.lattice.spins.hamiltonians.ising_transverse_field` and
`tachys.lattice.fermions.hamiltonians.hubbard`. The argument `nn` gives the
bonds by direction, one `((d1, d2), J)` entry per direction, where
$(d_1, d_2)$ is the displacement between the unit cells of the two sites: on
the square lattice, `((1, 0), J)` couples every site $(x, y)$ to $(x + 1, y)$,
and `((1, -1), J)` couples it to $(x + 1, y - 1)$. On a lattice with several
sites per cell, `((d1, d2), J, b_from, b_to)` connects sublattice `b_from` to
sublattice `b_to`, and `((d1, d2), J, b)` connects sublattice `b` to itself.

```python
from tachys.lattice.lattice_database import square, honeycomb
from tachys.lattice.spins.hamiltonians.heisenberg import heisenberg_hamiltonian
from tachys.lattice.fermions.hamiltonians.hubbard import hubbard_hamiltonian

# the J1-J2 model above
J1, J2 = 1.0, 0.5
H = heisenberg_hamiltonian(square(shape=(4, 4)), nn=[
    ((1, 0), J1), ((0, 1), J1),         # right, up
    ((1, 1), J2), ((1, -1), J2),        # up and right, down and right
])

# honeycomb lattice: every bond joins sublattice 0 (A) to sublattice 1 (B)
H = heisenberg_hamiltonian(honeycomb(shape=(4, 4)), nn=[
    ((0, 0), 1.0, 0, 1), ((-1, 0), 1.0, 0, 1), ((0, -1), 1.0, 0, 1),
])

# the Hubbard model above
t, U = 1.0, 8.0
H = hubbard_hamiltonian(square(shape=(4, 4)), nn=[((1, 0), t), ((0, 1), t)], U=U)
```

(operators-in-tachys)=

## Operators in tachys

Spin operators, in `tachys.lattice.spins.spin_operators`, act on a
`SpinState`:

| Operator | Definition | Name |
|---|---|---|
| `Sz(i)` | $S^z_i$ | spin component (diagonal) |
| `Sx(i)` | $S^x_i$ | spin component |
| `Sy(i)` | $S^y_i$ | spin component |
| `Splus(i)` | $S^+_i$ | raising operator |
| `Sminus(i)` | $S^-_i$ | lowering operator |
| `XYExchange(i, j)` | $S^+_iS^-_j + S^-_iS^+_j$ | spin flip-flop |

Fermionic operators, in `tachys.lattice.fermions.fermion_operators`, act on a
`FermionState`. The index `band` is 0 for $\uparrow$ and 1 for $\downarrow$:

| Operator | Definition | Name |
|---|---|---|
| `Cup_dag(i)`, `Cdn_dag(i)`, `C_dag(i, band)` | $c^\dagger_{i\uparrow}$, $c^\dagger_{i\downarrow}$, $c^\dagger_{i\sigma}$ | creation operator |
| `Cup(i)`, `Cdn(i)`, `C(i, band)` | $c_{i\uparrow}$, $c_{i\downarrow}$, $c_{i\sigma}$ | annihilation operator |
| `Nup(i)`, `Ndn(i)`, `N(i, band)` | $n_{i\uparrow}$, $n_{i\downarrow}$, $n_{i\sigma}$ | number operator (diagonal) |
| `HoppingUp(i, j)`, `HoppingDown(i, j)`, `Hopping(i, j, band)` | $\alpha\,c^\dagger_{i\sigma}c_{j\sigma} + \alpha^*\,c^\dagger_{j\sigma}c_{i\sigma}$ | hopping between $i$ and $j$ |
| `Sz_f(i)` | $\tfrac12\,(n_{i\uparrow} - n_{i\downarrow})$ | local spin (diagonal) |

Every operator takes a keyword argument `coupling`, a prefactor that defaults
to 1: `Sz(i, coupling=h)` is the same as `h * Sz(i)`. For the hopping
operators, the coupling is the amplitude $\alpha$, which may be complex.

The sites can also be arrays, one entry per term; every field, the coupling
included, then needs one entry per term. The built-in factories are written
this way.

## Writing an operator

A new elementary operator is a subclass of `_Operator`
(`tachys.lattice.operator.base`), like the ones above. Its fields hold the
parameters of one term, such as its sites, and `apply` returns, for every
configuration $x$ of the batch, the row of the term's matrix at $x$. The result
is a `DiagonalResult`, holding the diagonal matrix element
$\langle x|O|x\rangle$, or an `OffdiagonalResult`, which holds three arrays:

- `connected_states`, the configuration $x'$ with $\langle x|O|x'\rangle \neq 0$;
- `matrix_element`, the matrix element $\langle x|O|x'\rangle$;
- `mask`, which is `False` where the row of $x$ is empty.

Returning rows is what makes `compute_expectation` give
$\langle\psi|O|\psi\rangle$. The row of $S^+_i$, for example, is nonzero where
spin $i$ is up, and its connected configuration has that spin down, since
$\langle\uparrow|S^+|\downarrow\rangle = 1$.

tachys vectorizes `apply` over the terms. The permutation of two spins,
$P_{ij} = 2\,\mathbf{S}_i\cdot\mathbf{S}_j + \tfrac12$, swaps their values:

```python
import jax.numpy as jnp
from tachys.lattice.operator.base import _Operator, OffdiagonalResult

class Swap(_Operator):
    i: int
    j: int

    def apply(self, state):
        spins = state.spins                   # (N_mc, Ns)
        si, sj = spins[:, self.i], spins[:, self.j]
        swapped = spins.at[:, self.i].set(sj).at[:, self.j].set(si)
        n = spins.shape[0]
        return OffdiagonalResult(
            connected_states=state.replace(spins=swapped),
            mask=jnp.ones(n, dtype=bool),
            matrix_element=jnp.full(n, self.coupling),
        )
```

The mask is `True` everywhere: when the two spins are equal, the swap returns
$x$ itself with coefficient 1, which is not zero. `Swap(i, j)` combines with
the other operators as usual.
