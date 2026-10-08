# API Reference

The first sections follow the order of the {doc}`guide/index`. Each entry
gives the module it is imported from. Names that start with an underscore are
base classes, to subclass rather than call.

## Lattices

### `Lattice`

*`tachys.lattice.lattice`*

```python
class Lattice(lattice_vectors, basis, basis_frac, points, site_coords,
              cell_to_site, dist_matrix, L, Ns, nb, pbc)
```

The geometry of a finite two-dimensional cluster, stored as NumPy arrays.
Build it with `Lattice.create` or one of the
{ref}`built-in lattices <built-in-lattices>`, not from raw arrays.

Site `s` is sublattice `b` of the cell in row `i` (along `a2`) and column `j`
(along `a1`), with `s = (i * Lx + j) * nb + b`.

Build a lattice once and reuse it. Two lattices are equal only if they are the
same object, so states that carry two distinct copies have different pytree
structures: JAX compiles its functions again for each, and cannot combine
them. A lattice reaches jitted code as the static `lattice` field of a
`State`; do not pass it to a jitted function as an argument.

| Field | Description |
|---|---|
| `lattice_vectors` | `(2, 2)`: the rows are `a1` and `a2`. |
| `basis` | `(nb, 2)`: Cartesian positions of the basis sites in one cell. |
| `basis_frac` | `(nb, 2)`: the same positions in units of `a1` and `a2`. |
| `points` | `(Ns, 2)`: Cartesian position of every site. |
| `site_coords` | `(Ns, 3)`: `(i, j, b)` of every site. |
| `cell_to_site` | `(Ly, Lx, nb)`: the index of the site `(i, j, b)`. |
| `dist_matrix` | `(Ns, Ns)`: distances between sites, minimum image. |
| `L` | `(Lx, Ly)`: number of cells along `a1` and `a2`. |
| `Ns` | Number of sites. |
| `nb` | Number of sites per cell. |
| `pbc` | `(pbc_x, pbc_y)`: periodic boundaries along `a1` and `a2`. |

---

### `Lattice.create`

```python
Lattice.create(a1, a2, basis, shape, pbc_x=True, pbc_y=True)
```

A lattice from its primitive vectors, the positions of the basis sites in one
cell, and the number of cells.

| Parameter | Description |
|---|---|
| `a1`, `a2` | Primitive vectors, as 2-vectors. |
| `basis` | `(nb, 2)`: Cartesian positions of the basis sites. |
| `shape` | `(Lx, Ly)`: number of cells along `a1` and `a2`. |
| `pbc_x`, `pbc_y` | Periodic boundaries along `a1` and `a2`. Default `True`. |

**Returns** `Lattice`.

---

### `Lattice.bonds`

```python
lattice.bonds(delta, b_from=0, b_to=None)
```

The bonds from sublattice `b_from` in every cell $C$ to sublattice `b_to` in
the cell $C + \delta$, with $\delta = (d_1, d_2)$ counted in whole cells along
`a1` and `a2`; the offset within the cell comes from the sublattices only.
With open boundaries, the bonds that leave the cluster are dropped.

| Parameter | Description |
|---|---|
| `delta` | `(d1, d2)`: displacement between the cells of the two sites. |
| `b_from` | Sublattice of the first site. Default `0`. |
| `b_to` | Sublattice of the second site. Default: `b_from`. |

**Returns** `(src, dst)`: two `int64` arrays, with one entry per bond.

---

### `Lattice.bond_arrays`

```python
lattice.bond_arrays(deltas, b_from=0, b_to=None)
```

The bonds of several displacements: `bonds(delta, b_from, b_to)` for every
`delta` in `deltas`, concatenated.

**Returns** `(src, dst)`: two `jnp` integer arrays.

---

### `Lattice.neighbour_of`

```python
lattice.neighbour_of(site, delta, b_to=None)
```

The site reached from the cell of `site` by the displacement `delta`, on
sublattice `b_to` (default: that of `site`). Same convention as `bonds`.

**Returns** `int`: the site index, or `-1` if there is none.

---

### `Lattice.shells`

```python
lattice.shells(n_shells=None)
```

The pairs of sites grouped by distance, minimum image. Moves such as
`BondExchange` draw their pairs from these shells; Hamiltonians are built from
`bonds`, which distinguish the directions of the lattice.

| Parameter | Description |
|---|---|
| `n_shells` | Number of shells, closest first. Default `None`: all of them. |

**Returns** a list of `(distance, src, dst)`, one per shell, with every pair of
sites once (`src < dst`).

---

### `Lattice.retrieve_index`

```python
lattice.retrieve_index(c1, c2)
```

The site at the point $c_1\mathbf{a}_1 + c_2\mathbf{a}_2$, with `(c1, c2)`
including the offset of the basis site. Most code uses `bonds` or
`neighbour_of` instead.

**Returns** `int`: the site index, or `-1` if no site is there.

---

### `Lattice.plot`

```python
lattice.plot(filename=None)
```

Draws the sites with matplotlib, coloured by sublattice and labelled by index.
With `filename`, saves the figure there instead of showing it.

---

(built-in-lattices)=

### Built-in lattices

*`tachys.lattice.lattice_database`*

Every factory except `chain` takes `shape=(Lx, Ly)`, the number of cells along
`a1` and `a2`, and the periodic-boundary flags `pbc_x` and `pbc_y`.

| Factory | Geometry | `nb` |
|---|---|---|
| `chain(L, pbc=True)` | chain of `L` sites | 1 |
| `square(shape, pbc_x=True, pbc_y=True)` | square | 1 |
| `cylinder(shape, pbc_x=False, pbc_y=True)` | square, open along `a1` by default | 1 |
| `triangular(shape, pbc_x=True, pbc_y=True)` | triangular: `a1 = (1, 0)`, `a2` at 60° | 1 |
| `honeycomb(shape, pbc_x=True, pbc_y=True)` | triangular Bravais lattice, site B at `(a1 + a2) / 3` | 2 |
| `kagome(shape, pbc_x=True, pbc_y=True)` | triangular Bravais lattice | 3 |
| `shastry_sutherland(shape, pbc_x=True, pbc_y=True)` | square Bravais lattice, basis tilted by 10° | 4 |
| `plaquette(shape, pbc_x=True, pbc_y=True)` | square Bravais lattice, one 2 × 2 plaquette per cell | 4 |

On the honeycomb lattice, the three neighbours of an A site (sublattice 0) are
the B sites (sublattice 1) at the cell displacements `(0, 0)`, `(-1, 0)` and
`(0, -1)`.

## Operators

### `_Operator`

*`tachys.lattice.operator.base`*

```python
class _Operator(*, coupling=1.0)
```

The base class of the elementary operators. A subclass declares the fields of
one term, such as its sites, and implements `apply`. Operators combine into
Hamiltonians like the symbols of a formula ({doc}`guide/hamiltonian`).

| Member | Description |
|---|---|
| `coupling` | Prefactor of the matrix elements, keyword-only. Default `1.0`. |
| `apply(state)` | For every configuration $x$ of the batch, the row of one term at $x$, as a `DiagonalResult` or an `OffdiagonalResult`. Implemented by the subclasses. |
| `op(state)` | Applies every term of `op` to the batch; the result has a leading axis over the terms. |
| `A + B` | The sum. Terms of the same type and with the same static fields merge into one operator with array fields. |
| `c * A`, `A * c` | Multiplies the coupling by the Python number `c`. |
| `A * B` | The product $AB$. Both factors must have the same number of terms. |

The fields can be arrays with one entry per term; every field, the coupling
included, then needs one entry per term. Subtraction and negation are not
defined: write `A + (-1) * B`. Products of sums are not defined either: expand
them by hand.

With rows, the local estimator of any operator, Hermitian or not, averages to
$\langle\psi|O|\psi\rangle / \langle\psi|\psi\rangle$
(see `compute_expectation`).

---

### Results of `apply`

*`tachys.lattice.operator.base`*

```python
class DiagonalResult(matrix_element)
class OffdiagonalResult(connected_states, mask, matrix_element)
class DiagOffdiagResult(diagonal, offdiagonal)
```

The row of a term at every configuration $x$ of the batch:

| Field | Description |
|---|---|
| `matrix_element` | `(N_mc,)`: $\langle x \vert O \vert x\rangle$ for a `DiagonalResult`, $\langle x \vert O \vert x'\rangle$ for an `OffdiagonalResult`. |
| `connected_states` | A `State` holding the configuration $x'$ connected to each $x$. |
| `mask` | `(N_mc,)`, boolean: `False` where the row of $x$ is empty. |

Applying a sum of diagonal and off-diagonal terms gives a `DiagOffdiagResult`,
which holds one result of each kind.

---

### Spin operators

*`tachys.lattice.spins.spin_operators`*

They act on a `SpinState`, where $s_i = \pm 1$ is $S^z_i = \pm\tfrac12$.

| Operator | Definition |
|---|---|
| `Sz(site)` | $S^z_i$ (diagonal) |
| `Sx(site)` | $S^x_i$ |
| `Sy(site)` | $S^y_i$ |
| `Splus(site)` | $S^+_i$ |
| `Sminus(site)` | $S^-_i$ |
| `XYExchange(i, j)` | $S^+_iS^-_j + S^-_iS^+_j$ |

---

### Fermionic operators

*`tachys.lattice.fermions.fermion_operators`*

They act on a `FermionState`. `band` is 0 for $\uparrow$ and 1 for
$\downarrow$, and the mode of `(site, band)` has index `band * Ns + site`. The
Jordan–Wigner sign counts the occupied modes of lower index.

| Operator | Definition |
|---|---|
| `C(site, band)`, `Cup(site)`, `Cdn(site)` | $c_{i\sigma}$ |
| `C_dag(site, band)`, `Cup_dag(site)`, `Cdn_dag(site)` | $c^\dagger_{i\sigma}$ |
| `N(site, band)`, `Nup(site)`, `Ndn(site)` | $n_{i\sigma}$ (diagonal) |
| `Hopping(i, j, band)`, `HoppingUp(i, j)`, `HoppingDown(i, j)` | $\alpha\, c^\dagger_{i\sigma} c_{j\sigma} + \alpha^*\, c^\dagger_{j\sigma} c_{i\sigma}$, with $\alpha$ the coupling |
| `Sz_f(site)` | $\tfrac12\,(n_{i\uparrow} - n_{i\downarrow})$ (diagonal) |

## Hamiltonians

The factories take a `Lattice` and a list `nn` of bonds, with one entry per
direction:

- `((d1, d2), J)` couples sublattice 0 of every cell to sublattice 0 of the
  cell displaced by `(d1, d2)`;
- `((d1, d2), J, b)` does the same for sublattice `b`;
- `((d1, d2), J, b_from, b_to)` couples sublattice `b_from` to sublattice
  `b_to`.

The bonds are those of `lattice.bonds`, and `J` is the coupling of all of them
(the hopping amplitude, for the Hubbard model). The `*_square_pbc` factories
build the nearest-neighbour model on an `L × L` periodic square lattice, term
by term; they give the same Hamiltonian as the general factory with
`nn=[((1, 0), J), ((0, 1), J)]`.

### Heisenberg model

*`tachys.lattice.spins.hamiltonians.heisenberg`*

```python
heisenberg_hamiltonian(lat, nn)
heisenberg_square_pbc(L, J=1.0)
```

$$
H = \sum_{\langle i,j \rangle} J_{ij} \left[ S^z_i S^z_j + \tfrac{1}{2}(S^+_i S^-_j + S^-_i S^+_j) \right]
$$

---

### Transverse-field Ising model

*`tachys.lattice.spins.hamiltonians.ising_transverse_field`*

```python
ising_transverse_field_hamiltonian(lat, nn, h=1.0)
ising_transverse_field_square_pbc(L, J=1.0, h=1.0)
```

With the Pauli matrices $\sigma^\alpha = 2S^\alpha$,

$$
H = -\sum_{\langle i,j \rangle} J_{ij}\, \sigma^z_i \sigma^z_j - h \sum_i \sigma^x_i .
$$

---

### Hubbard model

*`tachys.lattice.fermions.hamiltonians.hubbard`*

```python
hubbard_hamiltonian(lat, nn, U)
hubbard_square_pbc(L, t=1.0, U=0.0)
```

$$
H = -\sum_{\langle i,j \rangle, \sigma} t_{ij} \left(c^\dagger_{i\sigma} c_{j\sigma} + \text{h.c.}\right)
    + U \sum_i n_{i\uparrow} n_{i\downarrow} ,
$$

with the hopping amplitudes $t_{ij}$ given by `nn`.

## Expectation values

### `compute_expectation`

*`tachys.lattice.operator.local_estimator`*

```python
compute_expectation(operator, wf, state, log_amps, optimize_mask=True, batch_expand=1)
```

The expectation value of an operator on a sample, from its local estimator

$$
O_L(x) = \sum_{x'} \langle x|O|x'\rangle\, \frac{\psi(x')}{\psi(x)} = \frac{\langle x|O|\psi\rangle}{\langle x|\psi\rangle} .
$$

On configurations drawn from $|\psi|^2$, the mean of $O_L$ estimates
$\langle\psi|O|\psi\rangle / \langle\psi|\psi\rangle$. For a Hermitian
operator the exact value is real, but the sample mean is complex in general:
its imaginary part is statistical noise. The chains are split among the
devices.

| Parameter | Description |
|---|---|
| `operator` | The operator. |
| `wf` | The `WaveFunction`. |
| `state` | The sample: a `State` of `N_mc` configurations. |
| `log_amps` | `(N_mc,)`: $\log\psi$ on `state`, as returned by `sample`. |
| `optimize_mask` | Evaluate $\psi(x')$ only where the mask of the operator is `True`. Default `True`. |
| `batch_expand` | Batch size of these evaluations, in units of the number of chains per device; it must divide the number of connected configurations, `n_terms * N_mc_local`. Default `1`. |

**Returns** `(O_L, O_mean, O2_mean)`: the local estimator, `(N_mc,)`; its
mean; and the mean of $|O_L|^2$. For the Hamiltonian,
`O2_mean - abs(O_mean)**2` is the energy variance, zero on an eigenstate.

---

### `local_estimator`

*`tachys.lattice.operator.local_estimator`*

```python
local_estimator(operator, state, wf, log_amps, optimize_mask=True, batch_expand=1)
```

$O_L(x)$ on every configuration of `state`, on a single device: the building
block of `compute_expectation`, for code that already runs on one shard of the
batch. Note the order of `state` and `wf`, swapped with respect to
`compute_expectation`.

**Returns** `(N_mc,)` array.

## States

### `State`

*`tachys.lattice.state`*

```python
class State(*, lattice=None)
```

The base class of `SpinState` and `FermionState`, a `flax.struct.PyTreeNode`.
Its data fields hold one row per chain, and the constructor checks that they
have the same leading dimension. A state built without a `lattice` warns that
`Ns` is unavailable.

| Member | Description |
|---|---|
| `lattice` | The `Lattice`, keyword-only; a static field. |
| `Ns` | Number of sites, `lattice.Ns`. |
| `replace(**fields)` | A copy of the state with the given fields replaced. |

---

### `SpinState`

*`tachys.lattice.spins.spin_state`*

```python
class SpinState(spins, *, lattice=None)
```

Spin-½ configurations: `spins`, of shape `(N_mc, Ns)`, with $+1$ for
$\uparrow$ and $-1$ for $\downarrow$.

---

### `init_config_fixed_magn`

*`tachys.lattice.spins.spin_state`*

```python
init_config_fixed_magn(key, N, sz=0, N_mc=1)
```

`N_mc` random configurations of `N` spins with total magnetization
$S^z$ = `sz`: `N/2 + sz` spins up and `N/2 - sz` down, at random positions.

**Returns** an `int8` array of shape `(N_mc, N)`.

---

### `FermionState`

*`tachys.lattice.fermions.fermion_state`*

```python
class FermionState(occupations, Ne, Nbands=2, *, lattice=None)
```

Occupation-number configurations of `Ne` electrons.

| Field | Description |
|---|---|
| `occupations` | `(N_mc, Nbands * Ns)`, entries 0 and 1. Mode `band * Ns + site`: first the ↑ modes of all sites, then the ↓ modes. |
| `Ne` | Number of electrons; static. |
| `Nbands` | Number of bands; static. Default `2`, spin up and down. |

---

### `init_config_spinful`

*`tachys.lattice.fermions.fermion_state`*

```python
init_config_spinful(key, Ns, Ne, sz=0, N_mc=1, particle_hole=False)
```

`N_mc` random configurations of `Ne` electrons (`Ne` even) on `Ns` sites, with
`N_up = Ne/2 + sz` electrons ↑ and `N_down = Ne/2 - sz` electrons ↓. With
`particle_hole=True`, the ↓ band holds `Ns - N_down` particles instead: the
configuration after a particle-hole transformation of the ↓ electrons.

**Returns** `(occupations, N_up, N_down)`, with `occupations` an `int32` array
of shape `(N_mc, 2 * Ns)` and `N_down` the number of particles in the ↓ band.

---

### State arrays

*`tachys.lattice.state_array`*

Helpers for code that handles any `State`, such as moves and symmetrizations.

| Function | Description |
|---|---|
| `get_array(state)` | The configuration array: `state.spins`, `state.occupations`, or `state.array` for another subclass of `State` that defines it. |
| `replace_array(state, new_array)` | A copy of `state` with that array replaced. A subclass with an `array` property must also define `replace_array(self, new_array)`. |
| `get_n_mc_local(state)` | The leading dimension of `state`: the number of chains on this device inside `shard_map`, all of them outside. |
| `get_n_mc(state)` | The total number of chains, inside `shard_map`: `get_n_mc_local(state) * n_devices`. |

## Wavefunctions

### `WaveFunction`

*`tachys.wavefunction`*

```python
class WaveFunction(params, apply_fn, unravel_params_fn=None, dtype=jnp.float64)
```

The parameters of an ansatz together with its apply function, as a pytree.
`sample`, `compute_expectation` and the optimizers evaluate the wavefunction
only as `wf.apply_fn(wf.params, state)`, which returns $\log\psi(x)$ for every
configuration $x$ of the batch: the real part is $\log|\psi(x)|$, the
imaginary part the phase.

| Member | Description |
|---|---|
| `params` | The parameters, a pytree. |
| `apply_fn` | `(params, state) -> log_psi`, typically `model.apply`; static. It is wrapped so that a single configuration, without the batch axis, is evaluated as a batch of one; the original is `wf.apply_fn.__wrapped__`. |
| `unravel_params_fn` | Maps a flat vector of parameters back to the pytree; static. Computed if not given. |
| `dtype` | Precision of sampling and of the local estimators: `sample` and `compute_expectation` cast the parameters and the floating-point fields of the state to it. Static; default `jnp.float64`. In float32, matmuls run at JAX's default precision, which is TF32 on recent NVIDIA GPUs. The optimizers have a `dtype` of their own. |
| `apply_gradients(grads, eta)` | A new `WaveFunction`, with parameters `params - eta * grads`. |
| `num_params` | Number of scalar parameters. |

---

### Ansätze

The ansätze are `flax.linen.Module`s. `params = model.init(key, state)` takes
a batch of configurations, and `model.apply(params, state)` returns one
log-amplitude per configuration, of shape `(batch,)`. Determinants carry their
sign as a phase, 0 or $\pi$; the RBM and the ViT learn a phase with
`complex=True`.

#### `SpinRBM`

*`tachys.lattice.ansatz.rbm`*

```python
class SpinRBM(hidden_units, dtype=jnp.float64, complex=False)
```

Restricted Boltzmann machine for spins, with `hidden_units` hidden units:

$$
\log\psi(s) = \sum_{k} \log\cosh z_k, \qquad z = W s + b .
$$

With `complex=True`, a second dense layer, applied to $z$, adds an imaginary
part: $z \to z + i\,(W' z + b')$. `dtype` is the dtype of the parameters.

#### `FermionRBM`

*`tachys.lattice.ansatz.rbm`*

```python
class FermionRBM(hidden_units)
```

Slater determinant with a backflow correction. The orbitals are parameters
$\phi_a(r)$, one per electron $a$ and mode $r$, corrected by $F_a(n, r)$,
which a two-layer network with `hidden_units` tanh units computes from the
whole configuration $n$:

$$
\psi(n) = \det\big[\phi_a(r_i) + F_a(n, r_i)\big]_{a,i=1}^{N_e},
$$

where $r_1, \dots, r_{N_e}$ are the occupied modes of $n$. A singular matrix
gives $\log\psi = -\infty$.

#### `log_cosh`

*`tachys.lattice.ansatz.rbm`*

```python
log_cosh(x)
```

$\log\cosh x$, for real or complex `x`, without overflow at large $|x|$.

#### `SpinViT`

*`tachys.lattice.ansatz.spin_vit`*

```python
class SpinViT(num_layers, d_model, num_heads, seq_len, b, complex=True,
              transl_invariant=False, two_dimensional=False, dtype=jnp.float64)
```

Vision transformer for spins; {doc}`resources/vit_wavefunction` builds a
variant of it step by step. The configuration is cut into patches, a linear
map sends each patch to a vector of size `d_model`, and an encoder of
`num_layers` blocks with factored attention mixes these vectors. The output
layer sums them and applies a dense layer and $\log\cosh$. The forward pass is
rematerialized (`nn.remat`), which saves memory in the backward pass.

| Field | Description |
|---|---|
| `num_layers` | Number of encoder blocks. |
| `d_model` | Size of the patch vectors; a multiple of `num_heads`. |
| `num_heads` | Number of attention heads. |
| `seq_len` | Number of patches: `Ns // b`, or `Ns // b**2` with `two_dimensional`. |
| `b` | Patch size: `b` consecutive sites, or `b × b` squares with `two_dimensional`. |
| `complex` | Adds a second output branch, for the phase. Default `True`. |
| `transl_invariant` | The attention between two patches depends only on their separation. Default `False`. |
| `two_dimensional` | Square patches on an `L × L` lattice; with `transl_invariant`, invariance under two-dimensional translations of the patches. Default `False`. |
| `dtype` | Dtype of the parameters. Default `jnp.float64`. |

Its building blocks, in the same module:

| Name | Description |
|---|---|
| `extract_patches1d(x, b)` | `(batch, N)` → `(batch, N // b, b)`: consecutive sites. |
| `extract_patches2d(x, b)` | `(batch, L * L)` → `(batch, (L // b)**2, b * b)`: `b × b` squares. |
| `Embed(d_model, b, dtype, two_dimensional=False)` | The patches, then one dense layer shared by all of them. |
| `OutputHead(d_model, dtype, complex)` | Sum over the patches, layer normalization, then a dense layer, layer normalization and $\sum \log\cosh$. With `complex`, a second dense branch gives the imaginary part. |

#### `FermionicTransformer`

*`tachys.lattice.ansatz.fermionic_transformer`*

```python
class FermionicTransformer(num_layers, d_model, num_heads, Ne, Ns, Nbands=2,
                           dtype=jnp.float64, transl_invariant=True, two_dimensional=True)
```

Slater determinant whose orbitals a transformer computes from the
configuration. The occupation of each site (empty, ↑, ↓ or both) is embedded
as a vector of size `d_model`, and the encoder mixes the vectors of the sites.
A linear map, different for each mode, sends the vector of a site to the
values of the `Ne` orbitals on its two modes, and the determinant is taken over
the occupied modes.

| Field | Description |
|---|---|
| `num_layers`, `d_model`, `num_heads` | As for `SpinViT`. |
| `Ne` | Number of electrons. |
| `Ns` | Number of sites; a perfect square with `two_dimensional`. |
| `Nbands` | Must be `2`. |
| `dtype` | Dtype of the parameters. Default `jnp.float64`. |
| `transl_invariant`, `two_dimensional` | As for `SpinViT`, with one site per patch. Default `True`. |

Its components, in the same module:

| Name | Description |
|---|---|
| `_log_det(A)` | $\log\det A$ as a complex number, $\log\vert\det A\vert + i\arg\det A$, with $-\infty$ for a singular matrix. |
| `compute_orbitals_fn(y, weights)` | `(batch, M, d)` and `(M, d, Ne)` → `(batch, M, Ne)`: the orbitals on the `M` modes. |
| `OutputHeadDet(d_model, Ne, Ns, dtype, Nbands=2)` | The output layer: the orbitals from the site vectors `y`, and the determinant over the occupied modes `R`, as `head(y, R)`. |

#### Transformer building blocks

*`tachys.lattice.ansatz.transformer.attention`*, *`tachys.lattice.ansatz.transformer.encoder`*

```python
class FactoredAttention(d_model, num_heads, seq_len, dtype,
                        transl_invariant=False, two_dimensional=False)
class EncoderBlock(d_model, num_heads, seq_len, dtype,
                   transl_invariant=False, two_dimensional=False)
class Encoder(num_layers, d_model, num_heads, seq_len, dtype,
              transl_invariant=False, two_dimensional=False)
```

The encoder of `SpinViT` and `FermionicTransformer`. Each module maps an array
of shape `(batch, seq_len, d_model)` to one of the same shape.

- `FactoredAttention` is multi-head attention whose weights are parameters,
  independent of the input: head $h$ returns $\alpha_h V_h \mathbf{x}$, and a
  dense layer mixes the heads. With `transl_invariant`, each head learns one
  row of $\alpha_h$ and the other rows are its cyclic shifts, so the weight
  between two positions depends only on their separation. With
  `two_dimensional` as well, the positions form a `√seq_len × √seq_len` grid,
  shifted along both axes.
- `EncoderBlock` applies the attention, then a two-layer feed-forward network
  (hidden size `4 * d_model`, GELU), each after a layer normalization and with
  a skip connection.
- `Encoder` stacks `num_layers` blocks.

---

### Sign rules

*`tachys.lattice.spins.sign_rules`*

A known sign structure, added to the log-amplitude as a phase, so that the
network learns only a positive amplitude ({doc}`guide/wavefunction`).

| Function | Description |
|---|---|
| `add_sign_rule(sign_fn, apply_fn, L)` | Wraps `apply_fn` into `(params, state) -> apply_fn(params, state) + sign_fn(state.spins, L)`. |
| `MSR_log_phase_square(spins, L)` | Marshall sign rule of the square lattice: $i\pi N_\downarrow^A$, with $A$ the sites of even row + column. |
| `MSR_log_phase_chain(spins, L)` | Marshall sign rule of a chain of `L` sites: $i\pi N_\downarrow^A$, with $A$ the even sites. |
| `triangular_classical_log_phase(spins, L)` | 120° rule of the triangular lattice: $i\tfrac{2\pi}{3}\sum_{i\,\downarrow} c_i$, with $c_i$ = (row − column) mod 3. |

For the square and triangular lattices, `L` is the shape `(Lx, Ly)` of the
cluster, such as `lattice.L`, or an `int` for an `L × L` cluster; the sites are
numbered as in `Lattice`. Each function returns a complex array of shape
`(batch,)`.

---

### Symmetries

*`tachys.lattice.symmetries`*

Wrappers that project a wavefunction onto a symmetry sector. Each takes an
apply function `(params, state) -> log_psi` and returns one with the same
signature.

| Function | Result |
|---|---|
| `symmetrize_wf(apply_fn, perms, sector_chars=None)` | $\log \sum_g \chi_g\, \psi(g\sigma)$, over the site permutations `perms` |
| `singlet_symm(apply_fn)` | $\log[\psi(\sigma) + \psi(-\sigma)]$: even under the global spin flip |
| `spin_flip_symm_f(apply_fn, p=1)` | For spinful fermions, the sector `p = ±1` of the spin flip $e^{-i\pi S^y}$ |
| `time_reversal(apply_fn)` | $\log[\psi + \psi^*] = \log 2\,\mathrm{Re}\,\psi$: a real wavefunction |

`symmetrize_wf` evaluates the network once per permutation. `perms` has shape
`(M, W)`, with `W` the width of the configuration array: `Ns` for spins, and
`2 * Ns` for fermions, where `expand_perm(perms, 2)` extends site permutations
to both spin species; the fermionic sign of each permutation is included.
`sector_chars` gives characters $\chi_g = e^{i\pi c_g}$ through the real
numbers $c_g$ (0 or 1 for $\chi_g = \pm 1$); by default $\chi_g = 1$.

`spin_flip_symm_f` requires $N_\uparrow = N_\downarrow$; `p=1` selects even
total spin, including the singlets, and `p=-1` odd total spin.

Permutation helpers, in the same module:

| Function | Description |
|---|---|
| `expand_perm(perm, n_bands)` | `(..., Ns)` → `(..., n_bands * Ns)`: the same site permutation in every band. |
| `invert_perm(perms)` | The inverse permutations. |
| `combine_perm_groups(perms_a, perms_b, chars_a=None, chars_b=None)` | The `M1 * M2` products `perms_b[b][perms_a[a]]` of two groups and their characters, to symmetrize over both in one `symmetrize_wf` call. Returns `(perms, chars)`. |
| `fermionic_sign(perm)` | The sign of a permutation, $\pm 1$. |
| `sign_permutation(config, perm_inv)` | The fermionic sign that a site permutation, given by its inverse, produces on the single configuration `config`. |

---

### Lattice symmetries

*`tachys.lattice.lattice_symmetries`*

The symmetry operations of a finite cluster, as site permutations `perm`, with
`perm[s]` the image of site `s`. `config[..., perm]` is the configuration
transformed by the inverse operation; since a projector sums over the whole
group, this does not matter for `symmetrize_wf`.

| Function | Description |
|---|---|
| `translation_group(lat)` | The translations of the cluster by whole cells, along its periodic directions. Returns `(perms, shifts)`: an `(Nt, Ns)` array of permutations, and the shift `(n1, n2)` of each, in cells along `a1` and `a2`. |
| `momentum_phases(lat, m)` | The characters $e^{-i\mathbf{k}\cdot\mathbf{R}_t}$ of these translations, in the same order, for $\mathbf{k} = 2\pi\,(m_1/L_x,\, m_2/L_y)$. The projector $\frac{1}{N_t}\sum_t e^{-i\mathbf{k}\cdot\mathbf{R}_t}\,T_t$ selects the states with $T_t \vert\psi\rangle = e^{i\mathbf{k}\cdot\mathbf{R}_t} \vert\psi\rangle$. |
| `point_group(lat, center=(0.0, 0.0), n_candidates=12)` | The rotations and reflections about `center` that map the cluster onto itself, as a list of `SymOp`: $C_{4v}$ on a square cluster, $C_{6v}$ on a compatible triangular one, a subgroup otherwise. The candidate angles are multiples of $360°$ / `n_candidates`. |

```python
class SymOp(name, matrix, perm)
```

A point-group operation: its name, such as `"C4"` or `"σv(0°)"`, its
$2 \times 2$ matrix, and its site permutation.

When the rotation centre of the group is not a site of sublattice 0, as for
the centre of a hexagon of the kagome lattice, pass it as `center`; otherwise
only the operations that fix the origin are found.

## Monte Carlo

### `sample`

*`tachys.montecarlo`*

```python
sample(nsweeps, state, action, key, wf)
```

Runs `nsweeps * Ns` Metropolis–Hastings steps on every chain. At each step,
`action` proposes a configuration $x'$, accepted with probability
$\min\big(1,\, |\psi(x')/\psi(x)|^2\, q(x|x')/q(x'|x)\big)$. The chains are
split among the devices.

| Parameter | Description |
|---|---|
| `nsweeps` | Number of sweeps, of `Ns` steps each. |
| `state` | The `N_mc` current configurations. |
| `action` | The move. |
| `key` | One PRNG key per chain, `jax.random.split(key, N_mc)`. |
| `wf` | The `WaveFunction` to sample. |

**Returns** `(state, log_amps, acceptance)`: the new configurations, their
log-amplitudes, and the fraction of accepted proposals, with one entry per move
of the action.

---

### `mc_step`

*`tachys.montecarlo`*

```python
mc_step(state, key, action, wf, log_amps, optimize_mask=True, batch_expand=0.25)
```

One Metropolis–Hastings step on every chain, on a single device: the building
block of `sample`, for custom samplers. `log_amps` holds $\log\psi$ of the
current configurations. With `optimize_mask`, the wavefunction is evaluated
only on the proposals that `action` allows, in batches of
`batch_expand * N_mc`.

**Returns** `(state, key, log_amps, accepted, action_id)`: `accepted` is a
boolean array of shape `(N_mc,)`, and `action_id` the index of the move used
on each chain (0 for a single move).

---

### `_BaseAction`

*`tachys.montecarlo`*

```python
class _BaseAction()
```

The base class of the moves, a `flax.struct.PyTreeNode`. A subclass implements
`__call__(key, state)`, which receives one PRNG key per chain and the batch of
configurations, and returns

- `new_state`, the proposed configurations;
- `allowed_move`, one boolean per chain: `False` rejects the proposal without
  evaluating the wavefunction;
- `log_prob_correction`, $\log[q(x|x')/q(x'|x)]$ for each chain, or `0.0` for
  a symmetric proposal.

A fourth value, the index of the move used on each chain, is optional.
`n_actions` is the number of moves, 1 for a single one.

---

### `CompositeAction`

*`tachys.montecarlo`*

```python
class CompositeAction(actions, probs)
```

At each step and on each chain, applies the move `actions[k]` with probability
`probs[k]`; `probs` must sum to 1. `sample` then returns one acceptance rate
per move.

---

### Moves

| Move | Module | Proposal |
|---|---|---|
| `BondExchange.create(lattice, max_dist=1, Nbands=1)` | `tachys.lattice.bond_exchange` | swap the values of the two sites of a bond |
| `FermionSpinExchange.create(lattice, max_dist=1)` | `tachys.lattice.fermions.fermion_action` | swap the spins of two singly occupied sites of a bond |
| `SpinFlip()` | `tachys.lattice.spins.spin_action` | flip one spin |
| `BondFlip.create(lattice, deltas, b_from=0, b_to=None)` | `tachys.lattice.spins.spin_action` | flip both spins of a bond |

- `BondExchange` draws a band, then a bond whose two sites differ in that band,
  and swaps their values: two antiparallel spins, or, with `Nbands=2` for
  fermions, an electron and an empty site of the same spin. It conserves $S^z$,
  or $N_\uparrow$ and $N_\downarrow$. The bonds are the pairs of sites in the
  first `max_dist` distance shells (`lattice.shells`), and the proposal
  probability is corrected for the number of valid bonds before and after the
  move.
- `FermionSpinExchange` draws a bond of the first `max_dist` shells. If one
  site holds only an ↑ electron and the other only a ↓ electron, it swaps their
  spins; otherwise the proposal is rejected. It moves no charge.
- `SpinFlip` flips a spin drawn at random, changing $S^z$ by one.
- `BondFlip` flips both spins of a bond drawn among `lattice.bonds(delta, b_from,
  b_to)` for every `delta` in `deltas`. It conserves the parity of
  $N_\uparrow$.

`exchange_spins(spins, id1, id2)`, in `tachys.lattice.spins.spin_action`,
swaps two entries of a single configuration.

## Optimizers

### Interface and shared fields

*`tachys.optimizer`*

`SR`, `SPRING` and `MARCH` are called in the same way,

```python
opt_state = optimizer.init(wf.params)
updates, opt_state = optimizer(E_L, opt_state, state, wf)
wf = wf.apply_gradients(updates, lr)
```

and share the fields of their base class, `_BaseOptimizer`
(`tachys.optimizer.optimizers`):

| Field | Description |
|---|---|
| `diag_shift` | The shift $\lambda$ added to the diagonal of the kernel. |
| `mode` | `"complex"`: the real and imaginary parts of $\log\psi$ enter as separate rows, and the phase is optimized. `"real"`: only $\log\vert\psi\vert$ enters, which is exact when the phase does not depend on the parameters. Static. |
| `nbatches` | Number of chunks in which each device evaluates the Jacobian, to limit memory. Static; default `1`. |
| `dtype` | Precision of the network evaluations in the update: the Jacobians, their contraction into the kernel, and the map back to parameter space. Static; default `None`, the dtype of the parameters. With `jnp.float32` (or `"float32"`), they run in single precision, with full-precision matmuls rather than TF32; the kernel and its solve stay in double precision. Single precision shifts the eigenvalues of the kernel by about $10^{-8}$ of the largest one: with a smaller `diag_shift`, the solve can fail, and a failed solve gives a zero step. |
| `kernels` | Where the kernel is built and solved. `"default"`: the whole kernel on every device. `"distributed"`: each device keeps its own rows, and JAXMg solves the system across the GPUs, one per process, so that no device holds the whole kernel. With a single device, `"distributed"` falls back to `"default"`, with a warning. Static; default `"default"`. |

The parameters must be real in both modes; a complex parameter gets a zero
update. `optimizer(E_L, opt_state, state, wf, weights=None)` also takes
per-sample importance weights, see {ref}`reweighted-estimators`.

The formulas below use the notation of {doc}`guide/optimization`: $\bar O$ is
the $N_{mc} \times P$ matrix of the centred log-derivatives
$\partial_{\theta_k}\log\psi(x)$, divided by $\sqrt{N_{mc}}$, and
$\bar\varepsilon$ the vector of the centred local energies, scaled by
$2/\sqrt{N_{mc}}$.

---

### `SR`

*`tachys.optimizer`*

```python
class SR(diag_shift, mode, nbatches=1, dtype=None, kernels="default")
```

Stochastic reconfiguration:

$$
\delta\theta = \bar O^T \big(\bar O\,\bar O^T + \lambda I\big)^{-1}\,\bar\varepsilon ,
$$

by a Cholesky solve of the $N_{mc} \times N_{mc}$ kernel ($2N_{mc} \times 2N_{mc}$
in `"complex"` mode) and a vector-Jacobian product. `init` returns an empty
`SRState()`.

---

### `SPRING`

*`tachys.optimizer`*

```python
class SPRING(diag_shift, mode, nbatches=1, dtype=None, kernels="default", *, mu=0.9)
```

SR with momentum `mu`, from Goldshlager, Abrahamsen & Lin,
["A Kaczmarz-inspired approach to accelerate the optimization of neural network
wavefunctions"](https://doi.org/10.1016/j.jcp.2024.113351), *Journal of
Computational Physics* (2024). The previous update $\delta\theta_{t-1}$ enters
both the right-hand side and the result:

$$
\delta\theta_t = \bar O^T \big(\bar O\,\bar O^T + \lambda I\big)^{-1}
\big(\bar\varepsilon - \mu\,\bar O\,\delta\theta_{t-1}\big) + \mu\,\delta\theta_{t-1} .
$$

`init` returns `SPRINGState(old_updates)`, the previous update, zero at the
start.

---

### `MARCH`

*`tachys.optimizer`*

```python
class MARCH(diag_shift, mode, nbatches=1, dtype=None, kernels="default", *, mu=0.95, beta=0.995)
```

SPRING with a rescaling of each parameter, in the spirit of Adam's second
moment, from Gu et al., ["Solving the Hubbard model with neural quantum
states"](https://doi.org/10.1038/s41467-026-74028-6), *Nature Communications*
(2026). $V$ is a running average, with decay rate `beta`, of the squared change
of the update between consecutive steps, $\hat V$ its bias-corrected value, and
$D = \mathrm{diag}\big(1/(\sqrt{\hat V} + 10^{-8})\big)$:

$$
\delta\theta_t = D\,\bar O^T \big(\bar O D \bar O^T + \lambda I\big)^{-1}
\big(\bar\varepsilon - \mu\,\bar O\,\delta\theta_{t-1}\big) + \mu\,\delta\theta_{t-1} .
$$

`init` returns `MARCHState(old_updates, V, t)`: the previous update, $V$
(ones at the start), and the step counter.

---

### Learning-rate schedules

*`tachys.optimizer`*

| Function | Description |
|---|---|
| `linear_decay(eta0, eta_final, N_steps)` | Linear, from `eta0` at step 0 to `eta_final` at step `N_steps`, and constant afterwards. |
| `shifted_cosine_decay(init_value, decay_steps, min_value=None)` | Cosine decay from `init_value` to `min_value` in `decay_steps` steps; `min_value` defaults to `init_value / 10`. |

Each returns a function of the step number. The step is absolute, so a resumed
run, whose steps start at `start_step`, continues the schedule.

---

### Kernel functions

*`tachys.optimizer._kernels`*

The building blocks of the optimizers, for writing new ones. They run inside
`shard_map`.

| Function | Description |
|---|---|
| `compute_ntk(state, wf, mode, weights=None, V=None, nbatches=1, dtype=None)` | The centred kernel $N_{mc}\,\bar O\,\bar O^T$: `ntk_parallel_fn`, then `center_ntk`, then the optional scaling of rows and columns by $\sqrt{w}$. |
| `ntk_parallel_fn(state, wf, nbatches, mode, V=None, dtype=None)` | The kernel of the whole batch before centring, of shape `(N_mc, N_mc)`, or `(N_mc, N_mc, 2, 2)` in `"complex"` mode, with the Jacobian contractions split among the devices. `V` is the rescaling of MARCH. |
| `center_ntk(ntk, weights, state)` | Subtracts the row, column and global means; per system for a `FoundationState`. |
| `linear_solver_cholesky(ntk, eps, diag_shift, mode="complex")` | Solves $(K + \lambda I)\,x = \varepsilon$ by Cholesky decomposition. In `"complex"` mode, as a real system of twice the size, returning `[u, v]` with $x = u + iv$. A failed solve returns zeros. |
| `linear_solver_eigh(ntk, eps, diag_shift, mode="complex", rcond=1e-8, atol=0.0)` | The same system, by diagonalization: eigenvalues below `max(rcond * λ_max, atol)` are discarded, and `diag_shift` is added to the others. Used by `TDVP`. |
| `center_sr_solution(sr_solution, state, mode, weights)` | Centres the solution before it is mapped back to parameter space; per system for a `FoundationState`. |

## Training

### `train`

*`tachys.ground_state_training`*

```python
train(key, H, state, wf, optimizer, action, N_steps, lr_schedule, N_mc,
      wandb_run=None, log_callback_fn=None, skip_optimization=False, nsweeps=1,
      opt_state=None, start_step=0, estimator=None)
```

Runs the optimization loop of the {doc}`quickstart`, and prints at every step
the energy per site, its variance, the V-score, the acceptance rate and the
timings. It exits the program if the energy per site becomes NaN or leaves the
interval $[-100, 100]$.

| Parameter | Description |
|---|---|
| `key` | PRNG key. |
| `H` | The Hamiltonian. |
| `state` | The initial configurations; `state.Ns` normalizes the energy per site. |
| `wf` | The initial `WaveFunction`. |
| `optimizer` | `SR`, `SPRING` or `MARCH`. |
| `action` | The Monte Carlo move. |
| `N_steps` | Number of steps of this call. |
| `lr_schedule` | The learning rate as a function of the step. |
| `N_mc` | Number of chains. |
| `wandb_run` | A Weights & Biases run, on the master process only. `train` logs the learning rate, energy, variance, V-score and acceptance there at every step, and writes checkpoints to `wandb_run.dir/checkpoints`: every `wandb_run.config["checkpoint_every"]` steps (default `N_steps`) and at the last step, keeping the last `checkpoint_keep` (default 1). |
| `log_callback_fn` | A function `(state, wf, step) -> dict or None`, or a list of them, whose metrics are added to the log. Called on every process. |
| `skip_optimization` | Only sample and measure the energy, without updating the parameters. Default `False`. |
| `nsweeps` | Sweeps per step. Default `1`. |
| `opt_state` | The optimizer state to start from, for instance from a checkpoint. Default: `optimizer.init(wf.params)`. |
| `start_step` | The number of the first step, to resume a run: it offsets the learning-rate schedule, the logs and the checkpoints. `N_steps` still counts the steps of this call. |
| `estimator` | Replaces `compute_expectation`, to sample from a distribution other than $\vert\psi\vert^2$; see below. |

**Returns** `(key, state, wf, opt_state, history)`. `history` is a dict of
lists, with one entry per step, for the keys `"energy"` (per site),
`"variance_per_site"`, `"vscore"`, `"acceptance"` and `"lr"`.

(reweighted-estimators)=

#### Reweighted estimators

An `estimator` samples from a distribution other than $|\psi|^2$, and corrects
with importance weights. `train` calls it as

```python
eval_state, E_L, weights, e_mean, e2_mean, metrics = estimator(keys, H, wf, state, log_amps)
```

with `keys` one PRNG key per chain. It returns the configurations on which it
evaluated `E_L`, which the optimizer receives in place of `state`; the
per-sample `weights`, or `None`, whose overall scale does not matter; `e_mean`
and `e2_mean`, as `compute_expectation` does; and a dict of extra scalars to
print and log. The Markov chains still evolve from `state`, which the callbacks
and the checkpoints see. Weights are not supported for foundation models.
`tachys.experimental.blurred_sampling.BlurredEstimator` is an estimator of this
kind.

---

### `compute_observables`

*`tachys.ground_state_training`*

```python
compute_observables(key, N_steps, state, action, wf, N_mc, op_groups, nsweeps=1,
                    log_every=1, complex=False)
```

Measures operators along the Markov chain, with `wf` fixed: at each of the
`N_steps` steps, it samples, then evaluates every operator on the same batch.

| Parameter | Description |
|---|---|
| `op_groups` | A dict of named lists of operators, or a single list, which is named `"obs"`. |
| `log_every` | Print a line every `log_every` steps; `0` for none. Default `1`. |
| `complex` | Keep the complex means; by default, only their real parts. |

The other parameters are those of `train`.

**Returns** `(key, state, metrics)`: `metrics[name]` is an array of shape
`(N_steps, len(op_groups[name]))`, the mean of each operator at each step.

## Foundation models

The objects of {doc}`foundation_models`.

### Foundation states

*`tachys.lattice.foundation.foundation_state`*

```python
SpinFoundationState(spins=..., system_couplings=..., system_ids=..., n_systems=..., lattice=...)
FermionFoundationState(occupations=..., Ne=..., system_couplings=..., system_ids=...,
                       n_systems=..., lattice=...)
```

A `SpinState` or a `FermionState` whose samples come from several systems.
Pass the fields by keyword. The three extra fields come from `FoundationState`:

| Field | Description |
|---|---|
| `system_couplings` | `(N_mc, n_couplings)`: the couplings of the system of each sample. |
| `system_ids` | `(N_mc,)`, integers in `[0, n_systems)`: the system of each sample. |
| `n_systems` | Number of systems; static. |

---

### `combine_systems`

*`tachys.lattice.foundation.operators`*

```python
combine_systems(operators, n_mc_per_system)
```

One operator for a batch that mixes several systems. `operators` holds one
Hamiltonian per system, all made by the same factory with different couplings.
In the result, the samples `k * n_mc_per_system` to
`(k + 1) * n_mc_per_system - 1` see `operators[k]`: every coupling becomes an
array of shape `(n_terms, N_mc)`, with one column per sample.

**Returns** the combined operator. **Raises** `ValueError` if the operators do
not have the same structure.

---

### `extract_system_couplings`

*`tachys.lattice.foundation.operators`*

```python
extract_system_couplings(operator, atol=1e-8, rtol=1e-5)
```

The couplings of a combined operator that vary from sample to sample, as the
array `system_couplings` of the state. There is one column per distinct
coupling, compared with `jnp.allclose(..., atol=atol, rtol=rtol)`: the
couplings shared by all systems are dropped, and a parameter that appears with
two prefactors, such as $J$ and $J/2$, gives two columns.

**Returns** an array of shape `(N_mc, n_couplings)`, or `(N_mc, 0)` if no
coupling varies.

---

### `broadcast_coupling`, `concatenate_couplings`

*`tachys.lattice.foundation.operators`*

```python
broadcast_coupling(operator, n_mc_per_system)
concatenate_couplings(operators)
```

The two steps of `combine_systems`. `broadcast_coupling` repeats every coupling
over the samples of one system, from shape `(n_terms,)` to
`(n_terms, n_mc_per_system)`; `concatenate_couplings` joins operators of the
same structure along the sample axis, and raises `ValueError` otherwise.

---

### Foundation ansätze

*`tachys.lattice.ansatz.rbm_foundation`*

```python
class SpinFoundationRBM(hidden_units, dtype=jnp.float64, complex=False)
class FermionFoundationRBM(hidden_units)
```

`SpinRBM` and `FermionRBM` with `state.system_couplings` appended to the input
of their first dense layer: to the spins for `SpinFoundationRBM`, to the input
of the backflow network for `FermionFoundationRBM`, whose bare orbitals are the
same for all systems.

---

### Grouped sums and means

*`tachys.lattice.foundation.collectives`*

```python
grouped_sum(x, y, K, axis=0)
grouped_mean(x, y, K, axis=0, broadcast=False)
```

Sums and means of `x` along `axis` within the `K` groups labelled by the
integers `y`, over all devices: for code inside `shard_map`, where each device
holds part of the batch. The result has size `K` along `axis`. With
`broadcast=True`, `grouped_mean` returns instead, for every element, the mean
of its group, sharded like `x`, so that
`x - grouped_mean(x, y, K, broadcast=True)` centres each group. `K` must be a
Python integer.

## Real-time dynamics

*`tachys.dynamics`*

Real-time evolution (t-VMC) follows $i\,\partial_t|\psi\rangle = H|\psi\rangle$
within the variational manifold. By the time-dependent variational principle,
the real parameters evolve as

$$
S\,\dot\theta = \mathrm{Im}\,F, \qquad
S_{kl} = \mathrm{Re}\langle \Delta O_k^{*}\,\Delta O_l\rangle, \qquad
F_k = \langle \Delta O_k^{*}\,\Delta E_L\rangle ,
$$

with $\Delta$ the deviation from the sample mean. Imaginary time gives
$S\,\dot\theta = -\mathrm{Re}\,F$ instead, the direction of the SR step, since
$\nabla E = 2\,\mathrm{Re}\,F$. tachys solves the equation with the same
kernel as `SR`, with two differences. The kernel is singular, since the
centring gives it zero modes and its rank cannot exceed the number of
parameters: `TDVP` discards its smallest eigenvalues instead of adding a
diagonal shift. And a step is a Runge–Kutta step, which draws a new sample at
each stage. The ansatz must be complex, since the evolution creates a phase.

### `TDVP`

*`tachys.dynamics`*

```python
class TDVP(*, diag_shift=0.0, mode, nbatches=1, dtype=None, kernels="default", rcond=1e-8, atol=0.0)
```

The velocity $\dot\theta$, called like an optimizer:
`dtheta_dt, opt_state = tdvp(E_L, opt_state, state, wf)`. It returns the
physical time derivative, which the integrators apply as
$\theta + \Delta t\,\dot\theta$; `apply_gradients`, which subtracts, would
reverse time.

| Field | Description |
|---|---|
| `rcond` | Eigenvalues of the kernel below `rcond` times the largest are discarded. The main parameter of a t-VMC run. Default `1e-8`. |
| `atol` | Absolute lower bound of that cutoff. Default `0.0`. |
| `diag_shift` | Added to the eigenvalues that are kept. Default `0.0`. |
| `mode` | Must be `"complex"`. |
| `kernels` | Must be `"default"`: the eigenvalue solve needs the whole kernel on every device. |
| `nbatches`, `dtype` | As for the optimizers. |

`init` returns an empty `TDVPState()`.

---

### `evolve`

*`tachys.dynamics`*

```python
evolve(key, H, state, wf, tdvp, action, N_steps, dt, N_mc,
       integrator="rk4", t0=0.0, wandb_run=None, log_callback_fn=None,
       nsweeps=1, opt_state=None, start_step=0, tdvp_error_every=0,
       tdvp_error_rule="rect")
```

The real-time counterpart of `train`: advances `wf` by `N_steps` steps of size
`dt`, and prints at every step the time, the energy per site, its variance and
its drift since the start.

| Parameter | Description |
|---|---|
| `H` | The Hamiltonian, or a function `t -> H(t)` for a time-dependent one. `H(t)` may change the values of the couplings, not the structure of the operator; `evolve` checks this at the start. |
| `wf` | A complex `WaveFunction`, for instance the result of `train`. |
| `tdvp` | A `TDVP`; an optimizer raises an error. |
| `dt` | Time step. A step costs one sample and one solve per stage of the integrator. |
| `integrator` | `"rk4"` (default), `"heun"`, or an `ExplicitRK`. |
| `t0` | Time of the first step. Default `0.0`. |
| `nsweeps` | Sweeps per stage. Default `1`. A slow drift of the energy means that the chains lag behind the wavefunction: increase `nsweeps`. |
| `tdvp_error_every` | Measure the TDVP error every this many steps (see `TDVPError`); `0`, the default, never. |
| `tdvp_error_rule` | `"rect"` (default) or `"trapezoid"`, see `TDVPError`. |

The other parameters are those of `train`; `log_callback_fn` receives the
wavefunction at the start of each step and the batch sampled from it.

**Returns** `(key, state, wf, opt_state, history)`. `history` has, with one
entry per step, the keys `"t"`, `"energy"` (per site), `"energy_real"` (the
total), `"variance_per_site"` and `"acceptance"`; with `tdvp_error_every`, also
`"R2"` and `"tdvp_rate"`, and `"tdvp_error"`, the history of the `TDVPError`
measurements.

---

### Integrators

*`tachys.dynamics`*

```python
class ExplicitRK(name, c, A, b, order)
Heun()                       # second order, 2 stages
RK4()                        # fourth order, 4 stages
get_integrator(integrator)   # "heun", "rk4" or an ExplicitRK
```

An explicit Runge–Kutta scheme, defined by its Butcher tableau: the stage
times `c`, the lower-triangular coefficients `A` and the weights `b`.
`step(rhs, key, t, wf, state, dt)` advances one step and returns
`(key, wf, state, ks, auxes)`, with `rhs(key, t, wf, state)` returning
`(key, state, dtheta_dt, aux)`. The Markov chains carry over from one stage to
the next, and every stage draws a new sample.

---

### `tdvp_error_rate`

*`tachys.dynamics`*

```python
tdvp_error_rate(wf, state, E_L, dtheta_dt, mode="complex")
```

The residual of the TDVP equation for the velocity `dtheta_dt`, per unit time
squared,

$$
\frac{\delta s^2}{\delta t^2} = \mathrm{Var}(H) + \dot\theta^T S\,\dot\theta - 2\,\mathrm{Im}(F)^T\dot\theta ,
$$

where $\delta s$ is the Fubini–Study distance, after one step, between the
exact and the variational evolution. `wf` must hold the parameters at which
`E_L` was measured, before the step.

**Returns** a `TDVPErrorEstimate` with the fields `rate`
($\delta s^2/\delta t^2$), `var_H`, `quad` ($\dot\theta^T S\,\dot\theta$),
`force` ($2\,\mathrm{Im}(F)^T\dot\theta$), `ratio` (`force / (2 * quad)`, 1
when $\dot\theta$ solves the TDVP equation) and `decomposed`
(`var_H + quad - force`, equal to `rate` up to rounding).

---

### `TDVPError`

*`tachys.dynamics`*

```python
class TDVPError(rule="rect", prefix="tdvp")
```

Accumulates the integrated error

$$
\mathcal{R}^2(t) = \frac{1}{\sqrt{N_s}} \int_0^t \sqrt{\frac{\delta s^2}{\delta t^2}}\; dt'
$$

from `tdvp_error_rate` measurements; `evolve(..., tdvp_error_every=n)` builds
one and measures every `n` steps. `rule` extends each measurement to the steps
before the next one: `"rect"` holds it constant, `"trapezoid"` interpolates
linearly. $\sqrt{N_s}\,\mathcal{R}^2$ bounds the Fubini–Study angle between
the exact and the variational state. It does not include the error of the time
discretization, and it can only grow: the current `rate`, or `rate / var_H`,
shows better whether the state is drifting away now.

| Member | Description |
|---|---|
| `R2` | $\mathcal{R}^2$ at the last measurement. |
| `history` | Dict of lists, with one entry per measurement: `step`, `t`, `rate`, `R2`, `var_H`, `quad`, `force`, `ratio`. |
| `accumulate(step, t, Ns, est)` | Adds a `TDVPErrorEstimate` and returns the metrics to log. |
| `reset()` | Clears `R2` and `history`. |

## Exact diagonalization

*`tachys.lattice.exact_diag`*

For clusters small enough to hold the whole Hilbert space;
{doc}`guide/hamiltonian` has an example.

| Function | Description |
|---|---|
| `spins_hilbert_space(N, values=(-1, 1))` | All $2^N$ configurations of `N` spins, an array of shape `(2**N, N)`; `values` are the values for down and up. |
| `fermions_hilbert_space(Ns, Ne, Nbands=2)` | All configurations of `Ne` electrons on `Nbands * Ns` modes. |
| `exact_diag(state_full_hilbert, H, pack, k=1)` | The `k` lowest eigenvalues of `H`, in ascending order, and their eigenvectors, by sparse diagonalization. Returns `(eigenvalues, eigenvectors)`. |
| `build_sparse_hamiltonian(state_full_hilbert, H, pack)` | The sparse matrix of `H`, for instance for exact time evolution. Returns `(matrix, sorted_active)`, with `sorted_active` the labels of `pack` in the order of the rows. |

`state_full_hilbert` is a `State` holding every configuration of the basis,
and `pack` maps a `State` to one distinct integer per configuration. `H` must
have both diagonal and off-diagonal terms.

## Checkpointing

*`tachys.checkpoint`*

`train` and `evolve` write checkpoints when they are given a `wandb_run`, to
`wandb_run.dir/checkpoints`. Each checkpoint holds the parameters, the
optimizer state, the configurations and the PRNG key, and is labelled by the
number of steps done.

### `load_checkpoint`

```python
load_checkpoint(checkpoint_dir, state_template, opt_state_template, params_template=None, step=None)
```

Restores `(params, opt_state, state, key)`. The templates supply what is not
saved: the static fields of the state, such as its lattice, and the type of the
optimizer state, for instance `optimizer.init(wf.params)`. The arrays are
placed on the current devices, which may differ from those of the run that
saved them.

| Parameter | Description |
|---|---|
| `params_template` | The structure of the parameters. If omitted, they are restored as a plain dict. |
| `step` | The step to restore. Default: the latest. |

To resume a run:

```python
from tachys.checkpoint import get_last_step, load_checkpoint

step = get_last_step(checkpoint_dir)
params, opt_state, state, key = load_checkpoint(
    checkpoint_dir, state, optimizer.init(wf.params), params_template=wf.params,
)
wf = wf.replace(params=params)
key, state, wf, opt_state, history = train(
    key, H, state, wf, optimizer, action, N_steps - step, lr_schedule, N_mc,
    opt_state=opt_state, start_step=step,
)
```

---

### Other functions

| Function | Description |
|---|---|
| `get_last_step(checkpoint_dir)` | The latest saved step, or `None`. |
| `save_training_checkpoint(manager, step, key, state, params, opt_state, force=False)` | Saves a checkpoint with an orbax `CheckpointManager`; `force` saves outside the interval of the manager. Called on every process. |
| `build_checkpoint_manager(directory, save_interval_steps, max_to_keep)` | An orbax `CheckpointManager` that saves every `save_interval_steps` steps and keeps the last `max_to_keep` checkpoints. |
| `resolve_checkpoint_settings(wandb_run, N_steps, rank, MASTER)` | The directory, interval and number of checkpoints, read from `wandb_run` on the master process and sent to all processes. `(None, None, None)` without a run. |

(api-parallelism)=

## Parallelism

*`tachys.parallel`*

`sample`, `compute_expectation` and the optimizers split the chains among all
the devices of `jax.devices()`, so `N_mc` must be a multiple of their number.
Run one process per device, launched with `mpirun` or `srun`
({doc}`parallel`): the optimizers assign their work by process, and with
several devices in one process they return wrong updates.

Importing `tachys`

- enables 64-bit precision in JAX (`jax_enable_x64`);
- calls `jax.distributed.initialize()` when launched by `mpirun` or `srun` with
  more than one process;
- restricts `print` to the master process;
- prints the devices and the versions of JAX and Flax.

| Name | Description |
|---|---|
| `mesh` | A `jax.sharding.Mesh` over all devices, with the single axis `'i'`. |
| `n_devices` | Number of devices. |
| `rank` | Index of this process, `jax.process_index()`. |
| `MASTER` | Rank of the process that prints, logs and reads the configuration: `0`. |
| `all_unshard(pytree)` | Replicates every array of `pytree` on all devices. |
| `hard_shard(pytree)` | Inside `shard_map` over `'i'`, the part of every array that belongs to this device: the `axis_index('i')`-th of `n_devices` equal parts of its first axis. |
| `promote_to_pytree(f)` | Turns a function of one array into a function of a pytree of arrays. |

## Utilities

*`tachys.utils`*

| Function | Description |
|---|---|
| `same_treedef(tree1, tree2)` | Whether two pytrees have the same structure, node types and static fields included. |
| `same_treedef_and_avals(tree1, tree2)` | Whether, in addition, all leaves have the same shapes and dtypes. |
| `as_column(x)` | A 1-D array as a column, of shape `(N, 1)`; other arrays unchanged. |
