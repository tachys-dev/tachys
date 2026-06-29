# API Reference

## States

### `SpinState`

*`tachys.lattice.spins.spin_state`*

```python
class SpinState(spins, Ns)
```

Batched spin-½ configurations on a lattice. Extends `flax.struct.PyTreeNode`.

| Field | Type | Description |
|-------|------|-------------|
| `spins` | `jax.Array` | Shape `(batch, N_sites)`. Spin values in {−1, +1}. |
| `Ns` | `int` | Number of lattice sites. |

---

### `init_config_fixed_magn`

*`tachys.lattice.spins.spin_state`*

```python
init_config_fixed_magn(key, N, sz=0, N_mc=1)
```

Sample random spin configurations with a fixed total magnetization.

| Parameter | Type | Description |
|-----------|------|-------------|
| `key` | `PRNGKey` | JAX random key. |
| `N` | `int` | Number of spins. |
| `sz` | `int` | Target magnetization. Default `0`. |
| `N_mc` | `int` | Number of configurations to generate. |

**Returns** `jnp.ndarray`, shape `(N_mc, N)`, values in {−1, +1}.

---

### `FermionState`

*`tachys.lattice.fermions.fermion_state`*

```python
class FermionState(occupations, Ns, Ne, Nbands=2)
```

Batched fermionic occupation-number configurations. Extends `flax.struct.PyTreeNode`.
Modes are ordered as (site 0 ↑, site 0 ↓, site 1 ↑, site 1 ↓, …).
`Ne` and `Nbands` are static (non-pytree) fields.

| Field | Type | Description |
|-------|------|-------------|
| `occupations` | `jax.Array` | Shape `(batch, Ns × Nbands)`. Binary occupation numbers ∈ {0, 1}. |
| `Ns` | `int` | Number of lattice sites. |
| `Ne` | `int` | Total number of electrons. Must be fixed at construction. |
| `Nbands` | `int` | Number of bands. Default `2` (spin-up / spin-down). |

---

### `init_config_spinful`

*`tachys.lattice.fermions.fermion_state`*

```python
init_config_spinful(key, Ns, Ne, sz=0, N_mc=1, particle_hole=False)
```

Sample random spinful fermionic configurations with fixed particle number and
spin magnetization.

| Parameter | Type | Description |
|-----------|------|-------------|
| `key` | `PRNGKey` | JAX random key. |
| `Ns` | `int` | Number of lattice sites. |
| `Ne` | `int` | Number of electrons. Must be even. |
| `sz` | `int` | Spin-magnetization offset. Default `0`. |
| `N_mc` | `int` | Number of configurations. |
| `particle_hole` | `bool` | Apply particle-hole transformation to the spin-down band. |

**Returns** `(config, N_up, N_down)`. `config` has shape `(N_mc, 2·Ns)`.

---

## Operators

### `_Operator`

*`tachys.lattice.operator.base`*

```python
class _Operator(coupling=1.0)
```

Abstract base for all operators. Subclass it and implement `apply(state)`.
Calling an instance automatically wraps `apply` in `jax.vmap` over the batch axis.

| Member | Type | Description |
|--------|------|-------------|
| `coupling` | `float` | Scalar prefactor applied to all matrix elements. |
| `__call__(state)` | `State → result` | Vectorized application over the batch axis. |
| `apply(state)` | `State → result` | Per-sample application. Override in subclasses. |
| `__add__(other)` | `_Operator` | Returns `_OperatorSum`. |
| `__mul__(other)` | scalar or `_Operator` | Scalar: rescales coupling. Operator: returns `_OperatorMul`. |

---

### `Sz`

*`tachys.lattice.spins.spin_operators`*

```python
class Sz(site, coupling=1.0)
```

Diagonal spin-z operator. Returns `DiagonalResult` with element `0.5 · coupling · σ_z`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `site` | `int` | Lattice site index. |
| `coupling` | `float` | Prefactor. Default `1.0`. |

---

### `Splus`

*`tachys.lattice.spins.spin_operators`*

```python
class Splus(site, coupling=1.0)
```

Raising operator S⁺ = (σ_x + iσ_y)/2. Flips the spin at `site` from ↓ to ↑.
Returns `OffdiagonalResult`; `mask=False` when the spin is already ↑.

---

### `Sminus`

*`tachys.lattice.spins.spin_operators`*

```python
class Sminus(site, coupling=1.0)
```

Lowering operator S⁻ = (σ_x − iσ_y)/2. Flips the spin at `site` from ↑ to ↓.
Returns `OffdiagonalResult`; `mask=False` when the spin is already ↓.

---

### `XYExchange`

*`tachys.lattice.spins.spin_operators`*

```python
class XYExchange(i, j, coupling=1.0)
```

Two-body term S⁺ᵢS⁻ⱼ + S⁻ᵢS⁺ⱼ. Non-zero only when spins at sites `i` and `j`
are antiparallel. Returns `OffdiagonalResult`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `i` | `int` | First site. |
| `j` | `int` | Second site. |
| `coupling` | `float` | Prefactor. Default `1.0`. |

---

### `C`

*`tachys.lattice.fermions.fermion_operators`*

```python
class C(site, band=1, coupling=1.0)
```

Fermionic annihilation operator c_{i,σ}. Removes an electron at `site` in `band`
(0 = ↑, 1 = ↓), including the Jordan-Wigner fermionic sign.
Returns `OffdiagonalResult`.

Convenience subclasses: `Cup(site)` sets `band=0`; `Cdn(site)` sets `band=1`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `site` | `int` | Lattice site index. |
| `band` | `int` | Band index: 0 = spin-up, 1 = spin-down. |
| `coupling` | `float` | Prefactor. Default `1.0`. |

---

### `C_dag`

*`tachys.lattice.fermions.fermion_operators`*

```python
class C_dag(site, band=1, coupling=1.0)
```

Fermionic creation operator c†_{i,σ}. Adds an electron at `site` in `band`,
including Jordan-Wigner sign. Returns `OffdiagonalResult`.

Convenience subclasses: `Cup_dag(site)` and `Cdn_dag(site)`.

---

### `N`

*`tachys.lattice.fermions.fermion_operators`*

```python
class N(site, band=1, coupling=1.0)
```

Number operator n_{i,σ} = c†_{i,σ} c_{i,σ}. Returns `DiagonalResult`.

Convenience subclasses: `Nup(site)` (band=0) and `Ndn(site)` (band=1).

---

## Hamiltonians

### `heisenberg_square_pbc`

*`tachys.lattice.spins.hamiltonians.heisenberg`*

```python
heisenberg_square_pbc(L, J=1.0)
```

Heisenberg model on an L×L square lattice with periodic boundary conditions.
Sites are indexed row-major: site at (x, y) maps to `x·L + y`.

$$
H = J \sum_{\langle i,j \rangle} \left[ S^z_i S^z_j + \tfrac{1}{2}(S^+_i S^-_j + S^-_i S^+_j) \right]
$$

| Parameter | Type | Description |
|-----------|------|-------------|
| `L` | `int` | Linear dimension. Total sites N = L². |
| `J` | `float` | Exchange coupling. Positive = antiferromagnetic. |

**Returns** `_OperatorSum`.

---

### `hubbard_square_pbc`

*`tachys.lattice.fermions.hamiltonians.hubbard`*

```python
hubbard_square_pbc(L, t=1.0, U=0.0)
```

Hubbard model on an L×L square lattice with periodic boundary conditions.
Two bands (spin-up / spin-down) with nearest-neighbor hopping and on-site
Coulomb repulsion.

$$
H = -t \sum_{\langle i,j \rangle, \sigma} (c^\dagger_{i\sigma} c_{j\sigma} + \text{h.c.})
    + U \sum_i n_{i\uparrow} n_{i\downarrow}
$$

| Parameter | Type | Description |
|-----------|------|-------------|
| `L` | `int` | Linear dimension. Total sites N = L². |
| `t` | `float` | Hopping amplitude. |
| `U` | `float` | On-site Coulomb repulsion. |

**Returns** `_OperatorSum`.

---

## Exact diagonalization

### `exact_diag`

*`tachys.lattice.exact_diag`*

```python
exact_diag(state_full_hilbert, H, pack, k=1)
```

Build the sparse Hamiltonian matrix and compute its `k` lowest eigenvalues.

Internally: applies `H` to every basis state, assembles a COO sparse matrix
from diagonal and off-diagonal results, and calls `scipy.sparse.linalg.eigsh`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `state_full_hilbert` | `State` | Batch containing every basis state in the Hilbert space. |
| `H` | `callable` | Hamiltonian operator. Must return `DiagOffdiagResult`. |
| `pack` | `callable` | Maps a state batch to a 1-D integer index array. Must be injective. |
| `k` | `int` | Number of lowest eigenvalues to compute. Default `1`. |

**Returns** `(eigenvalues, eigenvectors)`.  
Shapes: `eigenvalues` is `(k,)` in ascending order; `eigenvectors` is `(n_states, k)`.

---

### `spins_hilbert_space`

*`tachys.lattice.exact_diag`*

```python
spins_hilbert_space(N, values=(-1, 1))
```

Generate the complete Hilbert space for `N` spin-½ sites as all 2^N basis states.

| Parameter | Type | Description |
|-----------|------|-------------|
| `N` | `int` | Number of spins. |
| `values` | `tuple` | `(down, up)`. Default `(-1, 1)`. Use `(0, 1)` for binary encoding. |

**Returns** `np.ndarray`, shape `(2^N, N)`.

---

### `fermions_hilbert_space`

*`tachys.lattice.exact_diag`*

```python
fermions_hilbert_space(Ns, Ne, Nbands=2)
```

Generate all valid fermionic occupation configurations: all ways to place `Ne`
electrons on `Ns × Nbands` modes.

| Parameter | Type | Description |
|-----------|------|-------------|
| `Ns` | `int` | Number of lattice sites. |
| `Ne` | `int` | Number of electrons. |
| `Nbands` | `int` | Number of bands. Default `2`. |

**Returns** `np.ndarray`, shape `(C(Ns·Nbands, Ne), Ns·Nbands)` of binary
occupation vectors.
