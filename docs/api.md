# API Reference

## States

### `SpinState`

*`tachys.lattice.spins.spin_state`*

```python
class SpinState(spins, *, lattice=None)
```

Batched spin-½ configurations on a lattice. Extends `State` (see below), which
extends `flax.struct.PyTreeNode`.

| Field | Type | Description |
|-------|------|-------------|
| `spins` | `jax.Array` | Shape `(batch, N_sites)`. Spin values in {−1, +1}. |
| `lattice` | `Lattice` | Inherited from `State`. Static (`pytree_node=False`) lattice metadata. Default `None`. |

`Ns` (number of lattice sites) is **not** a constructor field — it's a
read-only property inherited from `State`, computed as `self.lattice.Ns`.
Construct with `SpinState(spins=..., lattice=lattice)`, not `Ns=...`.

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
class FermionState(occupations, Ne, Nbands=2, *, lattice=None)
```

Batched fermionic occupation-number configurations. Extends `State` (see
below), which extends `flax.struct.PyTreeNode`.
Modes are ordered band by band: (site 0 ↑, site 1 ↑, …, site Ns−1 ↑, site 0 ↓, …, site Ns−1 ↓),
i.e. mode `band * Ns + site`.
`Ne` and `Nbands` are static (non-pytree) fields.

| Field | Type | Description |
|-------|------|-------------|
| `occupations` | `jax.Array` | Shape `(batch, Ns × Nbands)`. Binary occupation numbers ∈ {0, 1}. |
| `Ne` | `int` | Total number of electrons. Must be fixed at construction. |
| `Nbands` | `int` | Number of bands. Default `2` (spin-up / spin-down). |
| `lattice` | `Lattice` | Inherited from `State`. Static (`pytree_node=False`) lattice metadata. Default `None`. |

`Ns` (number of lattice sites) is **not** a constructor field — it's a
read-only property inherited from `State`, computed as `self.lattice.Ns`.
Construct with `FermionState(occupations=..., Ne=..., lattice=lattice)`, not
`Ns=...`.

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

### `State`

*`tachys.lattice.state`*

```python
class State(lattice=None)
```

Base class for all lattice configuration containers (e.g. `SpinState`,
`FermionState`). Extends `flax.struct.PyTreeNode`.

Enforces that every data leaf's leading axis is the MC-batch dimension
(`N_mc`, or `N_mc_local` under sharding): `__post_init__` checks that all
data leaves with ≥2 dimensions share the same leading-axis size, raising
`ValueError` otherwise. If constructed without a `lattice`, emits a `UserWarning`
(skipped during jit/vmap/scan retracing, where dynamic fields are tracers).

| Field | Type | Description |
|-------|------|-------------|
| `lattice` | `Lattice` | Static (`pytree_node=False`) lattice metadata. Default `None`. |

| Property | Type | Description |
|----------|------|--------------|
| `Ns` | `int` | `self.lattice.Ns`. |

---

### `get_n_mc_local`

*`tachys.lattice.state_array`*

```python
get_n_mc_local(state)
```

The batch size of `state` along axis 0 as it currently stands (`N_mc_local` if
`state` is a per-device shard, else the same as `get_n_mc`) — read from any one
data leaf, since every `State` subclass's data fields share the same leading
batch axis (`State.__post_init__` checks this).

| Parameter | Type | Description |
|-----------|------|-------------|
| `state` | `State` | State batch (or per-device shard). |

**Returns** `int`.

---

### `get_n_mc`

*`tachys.lattice.state_array`*

```python
get_n_mc(state)
```

The global Monte Carlo batch size, even when `state` is currently a per-device
shard inside a `shard_map` body (`get_n_mc_local(state) * n_devices`).

| Parameter | Type | Description |
|-----------|------|-------------|
| `state` | `State` | State batch (or per-device shard). |

**Returns** `int`.

---

### `get_array`

*`tachys.lattice.state_array`*

```python
get_array(state)
```

The per-walker physical array a `State` subclass wraps: `state.spins` for
`SpinState`, `state.occupations` for `FermionState`, or a custom `.array`
property for subclasses that are neither (e.g. a composite state combining
several physical fields).

| Parameter | Type | Description |
|-----------|------|-------------|
| `state` | `State` | State batch. |

**Returns** `jax.Array`.

**Raises** `TypeError` if `state`'s subclass implements neither pattern.

---

### `replace_array`

*`tachys.lattice.state_array`*

```python
replace_array(state, new_array)
```

Return a copy of `state` with its physical array replaced by `new_array`.
Subclasses that fall back on `.array` in `get_array` must also implement a
`replace_array(self, new_array)` method mirroring `.array`'s getter with the
actual, possibly multi-field, update logic.

| Parameter | Type | Description |
|-----------|------|-------------|
| `state` | `State` | State batch. |
| `new_array` | `jax.Array` | Replacement physical array, same shape as `get_array(state)`. |

**Returns** `State`. A copy of `state` with the array field(s) replaced.

**Raises** `TypeError` if `state`'s subclass implements neither pattern.

---

### Foundation states

`FoundationState` and its subclasses provide the per-sample bookkeeping needed to train a single
ansatz across many distinct physical systems at once — a "foundation model" that shares one
network across a Monte Carlo batch mixing samples from several Hamiltonians (e.g. different
couplings or system sizes). It is a cross-cutting extension of the ordinary `State` hierarchy
(`SpinState`, `FermionState`), not a separate lattice type: it is combined via multiple
inheritance with a physical `State` subclass, carrying which system each sample in the batch
belongs to and that system's coupling values.

#### `FoundationState`

*`tachys.lattice.foundation.foundation_state`*

```python
class FoundationState(system_couplings, system_ids, n_systems)
```

Per-sample bookkeeping for training one ansatz across many systems at once. Extends
`flax.struct.PyTreeNode`. Meant to be combined via multiple inheritance with a physical `State`
subclass rather than used on its own — see `SpinFoundationState` and `FermionFoundationState`.

| Field | Type | Description |
|-------|------|-------------|
| `system_couplings` | `jax.Array` | Shape `(N_mc, n_couplings)`. Hamiltonian couplings (e.g. J, h) of each sample's system — one row per sample, matching the leading batch dimension of the paired `State`'s physical array (e.g. `spins`/`occupations`), *not* `(n_systems, n_couplings)`. |
| `system_ids` | `jax.Array` | Shape `(N_mc,)`. Per-sample integer label in `[0, n_systems)` identifying which system each sample belongs to. |
| `n_systems` | `int` | Total number of distinct systems in the batch. Static (non-pytree) field. |

---

#### `SpinFoundationState`

*`tachys.lattice.foundation.foundation_state`*

```python
class SpinFoundationState(spins, Ns, system_couplings, system_ids, n_systems)
```

`SpinState` samples tagged with their originating system for foundation-model training.
Inherits fields from both `SpinState` and `FoundationState`.

---

#### `FermionFoundationState`

*`tachys.lattice.foundation.foundation_state`*

```python
class FermionFoundationState(occupations, Ns, Ne, Nbands, system_couplings, system_ids, n_systems)
```

`FermionState` samples tagged with their originating system for foundation-model training.
Inherits fields from both `FermionState` and `FoundationState`.

---

## Operators

### `_Operator`

*`tachys.lattice.operator.base`*

```python
class _Operator(coupling=1.0)
```

Abstract base for all operators. Subclass it and implement `apply(state)`, which
returns, for every configuration x of the batch, the row of the operator at x:
the configurations x' with ⟨x|O|x'⟩ ≠ 0 and those matrix elements. With this
convention the local estimator averages to ⟨ψ|O|ψ⟩ for any operator, Hermitian
or not. Calling an instance wraps `apply` in `jax.vmap` over the terms.

| Member | Type | Description |
|--------|------|-------------|
| `coupling` | `float` | Scalar prefactor applied to all matrix elements. |
| `__call__(state)` | `State → result` | Applies every term to the batch (vmap over the terms). |
| `apply(state)` | `State → result` | Row of one term at every configuration of the batch. Override in subclasses. |
| `__add__(other)` | `_Operator` | Merges into one batched operator when the type *and* all static (`pytree_node=False`) fields match; otherwise returns `_OperatorSum`. |
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

Raising operator S⁺ = (σ_x + iσ_y)/2. Returns `OffdiagonalResult` with the row
⟨x|S⁺|x'⟩ = 1, where x' is x with the spin at `site` lowered; `mask=False` where
that spin is ↓ in x.

---

### `Sminus`

*`tachys.lattice.spins.spin_operators`*

```python
class Sminus(site, coupling=1.0)
```

Lowering operator S⁻ = (σ_x − iσ_y)/2. Returns `OffdiagonalResult` with the row
⟨x|S⁻|x'⟩ = 1, where x' is x with the spin at `site` raised; `mask=False` where
that spin is ↑ in x.

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
class C(site, band, coupling=1.0)
```

Fermionic annihilation operator c_{i,σ}, with the Jordan-Wigner sign. Returns
`OffdiagonalResult` with the row ⟨x|c_{i,σ}|x'⟩, where x' is x with an electron
added at `site` in `band` (0 = ↑, 1 = ↓); `mask=False` where that mode is
occupied in x.

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
class C_dag(site, band, coupling=1.0)
```

Fermionic creation operator c†_{i,σ}, with the Jordan-Wigner sign. Returns
`OffdiagonalResult` with the row ⟨x|c†_{i,σ}|x'⟩, where x' is x with the
electron at `site` in `band` removed; `mask=False` where that mode is empty in x.

Convenience subclasses: `Cup_dag(site)` and `Cdn_dag(site)`.

---

### `N`

*`tachys.lattice.fermions.fermion_operators`*

```python
class N(site, band, coupling=1.0)
```

Number operator n_{i,σ} = c†_{i,σ} c_{i,σ}. Returns `DiagonalResult`.

Convenience subclasses: `Nup(site)` (band=0) and `Ndn(site)` (band=1).

---

### Foundation operators

Combine per-system Hamiltonians into a single foundation-model operator. A foundation model
shares one wave function across a Monte Carlo batch that mixes samples from several distinct
systems (see the *Foundation states* subsection under States, above). The Hamiltonian has to mix the same way:
instead of one coupling per term (shape `(n_terms,)`, shared by every sample), each term needs a
per-sample coupling (shape `(n_terms, N_mc)`) that supplies the right system's value for each
column of the batch — vmapping `_Operator.apply` over the term axis then peels each leaf
operator's `coupling` down to exactly `(N_mc,)`, which is already what every `apply`
implementation expects to combine elementwise with state-derived quantities.

#### `broadcast_coupling`

*`tachys.lattice.foundation.operators`*

```python
broadcast_coupling(operator, n_mc_per_system)
```

Broadcast every leaf operator's 1-D coupling to 2-D: `(n_terms,) -> (n_terms, n_mc_per_system)`.
Every sample drawn from `operator`'s system sees the same per-term coupling, so the new trailing
axis is a plain repeat, not a fresh value per sample. Structural fields (`site`, `i`, `j`, …) are
left untouched. Recurses into `_OperatorSum`/`_OperatorMul` trees.

| Parameter | Type | Description |
|-----------|------|-------------|
| `operator` | `_Operator` | Single-system operator (leaf, `_OperatorSum`, or `_OperatorMul`). |
| `n_mc_per_system` | `int` | Number of Monte Carlo walkers dedicated to this system. |

**Returns** `_Operator` of the same tree structure, with every leaf's `coupling` broadcast to
shape `(n_terms, n_mc_per_system)`.

---

#### `concatenate_couplings`

*`tachys.lattice.foundation.operators`*

```python
concatenate_couplings(operators)
```

Concatenate same-structure operators' couplings along `axis=1`. Each of `operators` must already
have 2-D `(n_terms, n_mc_per_system)` couplings (see `broadcast_coupling`) and share identical
tree structure (e.g. built from the same Hamiltonian template with different coupling values).

| Parameter | Type | Description |
|-----------|------|-------------|
| `operators` | `list of _Operator` | Operators to concatenate, one per system, all with the same tree structure and 2-D couplings. |

**Returns** `_Operator` whose `coupling` has shape `(n_terms, sum of n_mc_per_system)` — one
column per Monte Carlo sample across all systems.

**Raises** `ValueError` if the operators do not share identical tree structure.

---

#### `combine_systems`

*`tachys.lattice.foundation.operators`*

```python
combine_systems(operators, n_mc_per_system)
```

Combine per-system operators into one foundation-model operator, by broadcasting every leaf
operator's 1-D coupling out to `(n_terms, n_mc_per_system)` and concatenating those along the
sample axis. `operators` must all be built from the same template (identical term structure) but
with different coupling values, exactly as produced by e.g.
`[hubbard_square_pbc(L, U=U) for U in Us]`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `operators` | `list of _Operator` | Per-system operators, one per system, built from the same Hamiltonian template. |
| `n_mc_per_system` | `int` | Number of Monte Carlo walkers dedicated to each system. |

**Returns** `_Operator` whose `coupling` has shape `(n_terms, len(operators) · n_mc_per_system)`,
matching the leading batch dimension of the paired `FoundationState`.

---

#### `extract_system_couplings`

*`tachys.lattice.foundation.operators`*

```python
extract_system_couplings(operator, atol=1e-8, rtol=1e-5)
```

Recover the distinct, sample-varying couplings of a combined operator (see `combine_systems`) as
the compact `(N_mc, n_couplings)` summary `FoundationState.system_couplings` expects.

`operator` must already be combined: every leaf's `coupling` is 2-D, `(n_terms, N_mc)` — one row
per term, one column per Monte Carlo sample. Every row of every leaf is a candidate per-sample
coupling; duplicate rows across leaves (e.g. several leaves sharing one fixed hopping amplitude)
collapse to a single column (rows compared with `jnp.allclose`, kept in first-seen traversal
order), rows that differ within one leaf (e.g. J1/J2 shells concatenated into one leaf's coupling)
split apart, and columns constant across all `N_mc` samples (they don't distinguish systems) are
dropped.

| Parameter | Type | Description |
|-----------|------|-------------|
| `operator` | `_Operator` | A combined operator, as returned by `combine_systems`. |
| `atol` | `float` | Absolute tolerance for `jnp.allclose` row comparisons. Default `1e-8`. |
| `rtol` | `float` | Relative tolerance for `jnp.allclose` row comparisons. Default `1e-5`. |

**Returns** `jax.Array`, shape `(N_mc, n_couplings)`. Suitable for
`FoundationState.system_couplings`. If no coupling varies across the batch, returns shape
`(N_mc, 0)`.

---

### Local estimators

Evaluate the local estimator of an operator, `O_L(x) = Σ_{x'} ⟨x|O|x'⟩ ψ(x')/ψ(x)`, given a
wave function and a batch of configurations, and reduce it to global expectation values across a
sharded device mesh.

#### `local_estimator`

*`tachys.lattice.operator.local_estimator`*

```python
local_estimator(operator, state, wf, log_amps, optimize_mask=True, batch_expand=1)
```

Local estimator $O_L(x) = \sum_{x'} \langle x|O|x'\rangle\, \psi(x')/\psi(x)$.

Applies `operator` to `state` to get a `DiagonalResult`, `OffdiagonalResult`, or
`DiagOffdiagResult`; diagonal terms are summed directly, off-diagonal terms require evaluating the
wave function on every connected state and forming the amplitude ratio `ψ(x')/ψ(x)` (via
`exp(log ψ(x') − log ψ(x))`, with the exponent — not the final result — masked so inactive,
possibly-placeholder connections can't overflow to `inf`/`NaN`).

| Parameter | Type | Description |
|-----------|------|-------------|
| `operator` | `_Operator` or `_OperatorSum` | Operator to evaluate. |
| `state` | `State` | Batch of configurations, batch axis 0 of size `N_mc_local`. |
| `wf` | wave function | Object with `.apply_fn(params, state) -> (N_mc_local,)` log-amplitudes and a `.params` attribute. |
| `log_amps` | `jax.Array` | Shape `(N_mc_local,)`. Log-amplitudes of `state` under `wf`. |
| `optimize_mask` | `bool` | Skip guaranteed-zero (all-mask-False) batches of connected states via a `while_loop`, instead of evaluating every connection. Default `True`. |
| `batch_expand` | `float` | Batch-size scale factor used by the masked-evaluation path (only used when `optimize_mask=True`); `int(batch_expand · N_mc_local)` must divide `N_terms · N_mc_local`. Default `1`. |

**Returns** `jax.Array`, shape `(N_mc_local,)`. The local estimator `O_L`.

---

#### `compute_expectation`

*`tachys.lattice.operator.local_estimator`*

```python
compute_expectation(operator, wf, state, log_amps, optimize_mask=True, batch_expand=1)
```

Sharded expectation value of an operator. Shards `state` and `log_amps` across all devices,
evaluates `local_estimator` on each shard, then reduces to global statistics via `psum`. JIT-
compiled with `optimize_mask` and `batch_expand` as static arguments.

A foundation-model operator's `coupling` (see the *Foundation operators* subsection above),
when 2-D with shape `(n_terms, N_mc)`, is sharded along the `N_mc` axis to match `state`'s sharded
batch axis; all other operator fields, and non-foundation operators, are replicated across
devices.

| Parameter | Type | Description |
|-----------|------|-------------|
| `operator` | `_Operator` | Replicated across devices, except a foundation-model operator's `coupling` (`ndim == 2`, shape `(n_terms, N_mc)`), which is sharded along the `N_mc` axis. |
| `wf` | wave function | Replicated across devices. |
| `state` | `State` | Batch axis 0 sharded across devices. |
| `log_amps` | `jax.Array` | Shape `(N_mc_local,)`, sharded across devices. |
| `optimize_mask` | `bool` | Forwarded to `local_estimator`. Default `True`. |
| `batch_expand` | `int` | Forwarded to `local_estimator`. Default `1`. |

**Returns** `(O_L, O_mean, O2_mean)`:
- `O_L` — `jax.Array`, shape `(N_mc_local,)`, the local estimator, sharded.
- `O_mean` — scalar, global mean `⟨O⟩`.
- `O2_mean` — scalar, global mean `⟨|O|²⟩`.

---

## Hamiltonians

### `heisenberg_hamiltonian`

*`tachys.lattice.spins.hamiltonians.heisenberg`*

```python
heisenberg_hamiltonian(lat, nn)
```

Heisenberg Hamiltonian on a generic `Lattice`, assembled from arbitrary bond
specifications instead of a fixed periodic square geometry — the building
block behind `heisenberg_square_pbc` and behind custom lattices (triangular,
honeycomb, multiple coupling shells, …).

$$
H = \sum_{\langle i,j \rangle} J_{ij} \left[ S^z_i S^z_j + \tfrac{1}{2}(S^+_i S^-_j + S^-_i S^+_j) \right]
$$

| Parameter | Type | Description |
|-----------|------|-------------|
| `lat` | `Lattice` | Lattice object providing bond geometry via `lat.bonds`. |
| `nn` | `list` | Bond specifications, each pairing a cell displacement with its coupling: `((d1, d2), J_ij)` (source sublattice `b_from=0`), `((d1, d2), J_ij, b_from)`, or `((d1, d2), J_ij, b_from, b_to)`. List further shells (e.g. next-nearest-neighbour bonds) as additional entries. |

**Returns** `_OperatorSum`.

---

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

### `hubbard_hamiltonian`

*`tachys.lattice.fermions.hamiltonians.hubbard`*

```python
hubbard_hamiltonian(lat, nn, U)
```

Hubbard Hamiltonian on a generic `Lattice`, assembled from arbitrary bond
specifications instead of a fixed periodic square geometry — the building
block behind `hubbard_square_pbc` and behind custom lattices or bond-dependent
hopping amplitudes.

$$
H = -\sum_{\langle i,j \rangle, \sigma} t_{ij} \left(c^\dagger_{i\sigma} c_{j\sigma} + \text{h.c.}\right)
    + U \sum_i n_{i\uparrow} n_{i\downarrow}
$$

| Parameter | Type | Description |
|-----------|------|-------------|
| `lat` | `Lattice` | Lattice object providing bond geometry via `lat.bonds`. |
| `nn` | `list` | Bond specifications, each pairing a cell displacement with its hopping amplitude: `((d1, d2), t_ij)` (source sublattice `b_from=0`), `((d1, d2), t_ij, b_from)`, or `((d1, d2), t_ij, b_from, b_to)`. List further shells (e.g. next-nearest-neighbour hopping) as additional entries. |
| `U` | `float` | On-site Coulomb repulsion. |

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

### Ising model

#### `ising_transverse_field_hamiltonian`

*`tachys.lattice.spins.hamiltonians.ising_transverse_field`*

```python
ising_transverse_field_hamiltonian(lat, nn, h=1.0)
```

Transverse-field Ising Hamiltonian on a generic `Lattice`, in Pauli-matrix convention
(σ = 2S).

$$
H = -\sum_{\langle i,j \rangle} J_{ij}\, \sigma^z_i \sigma^z_j - h \sum_i \sigma^x_i
$$

| Parameter | Type | Description |
|-----------|------|-------------|
| `lat` | `Lattice` | Lattice object providing bond geometry via `lat.bonds`. |
| `nn` | `list` | Bond specifications, each pairing a cell displacement with its coupling: `((d1, d2), J_ij)` (source sublattice `b_from=0`), `((d1, d2), J_ij, b_from)`, or `((d1, d2), J_ij, b_from, b_to)`. List further shells (e.g. next-nearest-neighbour bonds) as additional entries. |
| `h` | `float` | Transverse field strength. Default `1.0`. |

**Returns** `_OperatorSum`.

---

#### `ising_transverse_field_square_pbc`

*`tachys.lattice.spins.hamiltonians.ising_transverse_field`*

```python
ising_transverse_field_square_pbc(L, J=1.0, h=1.0)
```

Transverse-field Ising model on an L×L square lattice with periodic boundary conditions, in
Pauli-matrix convention (σ = 2S). Sites are indexed row-major: `site(x, y) = x·L + y`. The 1-D
chain's critical point is at `J = h`.

$$
H = -J \sum_{\langle i,j \rangle} \sigma^z_i \sigma^z_j - h \sum_i \sigma^x_i
$$

| Parameter | Type | Description |
|-----------|------|-------------|
| `L` | `int` | Linear dimension. Total sites N = L². |
| `J` | `float` | Nearest-neighbour Ising coupling. Default `1.0`. |
| `h` | `float` | Transverse field strength. Default `1.0`. |

**Returns** `_OperatorSum`.

---

## Lattices

### `Lattice`

*`tachys.lattice.lattice`*

```python
class Lattice(lattice_vectors, basis, basis_frac, points, site_coords,
               cell_to_site, dist_matrix, L, Ns, nb, pbc)
```

Immutable lattice geometry. Extends `typing.NamedTuple`. Always build one via
`Lattice.create(...)`, never by calling the constructor with raw arrays directly.

A site is an integer triple `(i, j, b)`: `i` = cell index along `a2` (row), `j` =
cell index along `a1` (column), `b` = sublattice. `site_coords[s] = (i, j, b)` and
the reverse map `cell_to_site[i, j, b] -> s` (`-1` where absent) make every lookup
pure integer/modular arithmetic. `.bonds(delta, b_from, b_to)` connects sublattice
`b_from` in cell `C` to sublattice `b_to` in cell `C + delta`, for every cell `C`;
`delta` is a whole-cell displacement in `(a1, a2)` units — the intra-cell offset
comes only from `b_from`/`b_to`.

Equality and hashing are identity-based (`__eq__`/`__hash__` use `id(self)`), so a
`Lattice` can sit as static (`pytree_node=False`) metadata on a `State` without
JAX trying to hash or compare its numpy array fields.

A `NamedTuple` is an automatic JAX pytree, so passing a `Lattice` directly into a
jitted function makes JAX try to flatten these numpy arrays into leaves. Instead,
build the jnp bond-index arrays once with `.bond_arrays()` and pass only those
into traced code; treat `Lattice` itself as host-side metadata.

| Field | Type | Description |
|-------|------|-------------|
| `lattice_vectors` | `np.ndarray` | `(2, 2)`. Rows are `a1`, `a2`. |
| `basis` | `np.ndarray` | `(nb, 2)`. Cartesian positions of the basis atoms in one cell. |
| `basis_frac` | `np.ndarray` | `(nb, 2)`. Basis positions in `(a1, a2)` fractional units. |
| `points` | `np.ndarray` | `(Ns, 2)`. Cartesian coordinates of every site. |
| `site_coords` | `np.ndarray` | `(Ns, 3)`. `(i, j, b)` per site. |
| `cell_to_site` | `np.ndarray` | `(Ly, Lx, nb)`. Reverse lookup; `-1` where absent. |
| `dist_matrix` | `np.ndarray` | `(Ns, Ns)`. Minimum-image Euclidean distances (for observables/plotting only). |
| `L` | `tuple` | `(Lx, Ly)`, number of cells along `a1`/`a2`. |
| `Ns` | `int` | Total number of sites. |
| `nb` | `int` | Number of basis atoms per cell. |
| `pbc` | `tuple` | `(pbc_x, pbc_y)`. |

---

### `Lattice.create`

*`tachys.lattice.lattice`*

```python
Lattice.create(a1, a2, basis, shape, pbc_x=True, pbc_y=True)
```

Classmethod constructor. Builds a lattice from unit-cell vectors, a Cartesian
basis, and a shape, computing the site table and the minimum-image distance
matrix.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a1`, `a2` | `2-vector` | Primitive cell vectors. |
| `basis` | `(nb, 2) array` | Cartesian positions of the atoms in one cell. |
| `shape` | `(Lx, Ly)` | Number of cells along `a1` and `a2`. |
| `pbc_x`, `pbc_y` | `bool` | Periodic boundaries along `a1`/`a2`. Default `True`. |

**Returns** `Lattice`.

---

### `Lattice.retrieve_index`

*`tachys.lattice.lattice`*

```python
lattice.retrieve_index(c1, c2)
```

Low-level lookup: site index for target coefficients `(c1, c2)` in the `(a1, a2)`
basis, including any basis offset. Most code should use `bonds`/`neighbour_of`
instead of calling this directly.

| Parameter | Type | Description |
|-----------|------|-------------|
| `c1`, `c2` | `float` | Target coefficients in the `(a1, a2)` basis. |

**Returns** `int`. Site index, or `-1` if no site exists there (OBC out of range,
or the point does not coincide with any atom).

---

### `Lattice.bonds`

*`tachys.lattice.lattice`*

```python
lattice.bonds(delta, b_from=0, b_to=None)
```

Directed `(src, dst)` index pairs for a cell displacement `delta`. A bond
connects sublattice `b_from` in cell `C` to sublattice `b_to` in cell `C + delta`,
for every cell `C` for which the target exists. This is the primitive used to
assemble Hamiltonian bond lists.

| Parameter | Type | Description |
|-----------|------|-------------|
| `delta` | `(d1, d2)` | Cell displacement in `(a1, a2)` units (whole cells; the intra-cell offset comes from `b_from`/`b_to`, not `delta`). |
| `b_from` | `int` | Source sublattice. Default `0`. |
| `b_to` | `int` | Target sublattice. Default: same as `b_from`. |

**Returns** `(src, dst)`, `int64` arrays of equal length. Under OBC, bonds whose
target falls outside the lattice are dropped.

---

### `Lattice.neighbour_of`

*`tachys.lattice.lattice`*

```python
lattice.neighbour_of(site, delta, b_to=None)
```

Single site reached from `site`'s cell by cell displacement `delta`, landing on
sublattice `b_to` (default: same sublattice as `site`). Same convention as
`bonds`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `site` | `int` | Source site index. |
| `delta` | `(d1, d2)` | Cell displacement in `(a1, a2)` units. |
| `b_to` | `int` | Target sublattice. Default: same as `site`'s. |

**Returns** `int`. Neighbour site index, or `-1` if absent.

---

### `Lattice.bond_arrays`

*`tachys.lattice.lattice`*

```python
lattice.bond_arrays(deltas, b_from=0, b_to=None)
```

Concatenate several cell displacements into flat `jnp` int arrays `(src, dst)`,
ready to feed a jitted local energy. All displacements share `b_from`/`b_to`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `deltas` | `iterable of (d1, d2)` | Cell displacements to concatenate. |
| `b_from` | `int` | Source sublattice. Default `0`. |
| `b_to` | `int` | Target sublattice. Default: same as `b_from`. |

**Returns** `(src, dst)`, `jnp.ndarray` int arrays, concatenated over all `deltas`.

---

### `Lattice.shells`

*`tachys.lattice.lattice`*

```python
lattice.shells(n_shells=None)
```

Distance-shell neighbour lists, for correlation functions / structure factors.
**Not** for Hamiltonian construction — use `bonds()` there.

| Parameter | Type | Description |
|-----------|------|-------------|
| `n_shells` | `int` | Number of nearest distinct distances to return. Default `None` (all shells). |

**Returns** `list[(distance, src, dst)]`. Each ordered pair of sites at that
distance is counted once (`src[k] < dst[k]`).

---

### `Lattice.plot`

*`tachys.lattice.lattice`*

```python
lattice.plot(filename=None)
```

Scatter the sites, coloured by sublattice and labelled by index (via
`matplotlib`).

| Parameter | Type | Description |
|-----------|------|-------------|
| `filename` | `str` | If given, saves the figure to this path instead of calling `plt.show()`. |

**Returns** `None`.

---

### `chain`

*`tachys.lattice.lattice_database`*

```python
chain(L, pbc=True)
```

1-D chain of `L` sites, open along the (unused) second direction.

| Parameter | Type | Description |
|-----------|------|-------------|
| `L` | `int` | Number of sites. |
| `pbc` | `bool` | Periodic boundary conditions along the chain. Default `True`. |

**Returns** `Lattice`.

---

### `square`

*`tachys.lattice.lattice_database`*

```python
square(shape, pbc_x=True, pbc_y=True)
```

Square lattice, one site per cell.

| Parameter | Type | Description |
|-----------|------|-------------|
| `shape` | `(Lx, Ly)` | Number of cells along `a1`/`a2`. |
| `pbc_x`, `pbc_y` | `bool` | Periodic boundaries along each direction. Default `True`. |

**Returns** `Lattice`.

---

### `triangular`

*`tachys.lattice.lattice_database`*

```python
triangular(shape, pbc_x=True, pbc_y=True)
```

Triangular lattice, one site per cell, with `a1 = (1, 0)` and `a2` at 60° to `a1`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `shape` | `(Lx, Ly)` | Number of cells along `a1`/`a2`. |
| `pbc_x`, `pbc_y` | `bool` | Periodic boundaries along each direction. Default `True`. |

**Returns** `Lattice`.

---

### `honeycomb`

*`tachys.lattice.lattice_database`*

```python
honeycomb(shape, pbc_x=True, pbc_y=True)
```

Honeycomb lattice: the same triangular Bravais lattice as `triangular`, with a
2-site basis (A at the cell origin, B at the centroid `(a1+a2)/3`). Each A site
has 3 nearest-neighbour B sites, reached by cell displacements `(0,0)`, `(-1,0)`,
`(0,-1)` — useful for Kitaev-type Hamiltonians that assign each of these its own
bond-dependent operator.

| Parameter | Type | Description |
|-----------|------|-------------|
| `shape` | `(Lx, Ly)` | Number of cells along `a1`/`a2`. |
| `pbc_x`, `pbc_y` | `bool` | Periodic boundaries along each direction. Default `True`. |

**Returns** `Lattice`. `nb=2`.

---

### `kagome`

*`tachys.lattice.lattice_database`*

```python
kagome(shape, pbc_x=True, pbc_y=True)
```

Kagome lattice: triangular Bravais lattice with a 3-site basis.

| Parameter | Type | Description |
|-----------|------|-------------|
| `shape` | `(Lx, Ly)` | Number of cells along `a1`/`a2`. |
| `pbc_x`, `pbc_y` | `bool` | Periodic boundaries along each direction. Default `True`. |

**Returns** `Lattice`. `nb=3`.

---

### `cylinder`

*`tachys.lattice.lattice_database`*

```python
cylinder(shape, pbc_x=False, pbc_y=True)
```

Square lattice on a cylinder: periodic along `y` (rows) and open along `x`
(columns) by default.

| Parameter | Type | Description |
|-----------|------|-------------|
| `shape` | `(Lx, Ly)` | Number of cells along `a1`/`a2`. |
| `pbc_x` | `bool` | Periodic along `x`. Default `False`. |
| `pbc_y` | `bool` | Periodic along `y`. Default `True`. |

**Returns** `Lattice`.

---

### `shastry_sutherland`

*`tachys.lattice.lattice_database`*

```python
shastry_sutherland(shape, pbc_x=True, pbc_y=True)
```

Shastry-Sutherland lattice: square Bravais lattice (`a1=(1,0)`, `a2=(0,1)`) with
a 4-site basis arranged around a 10° tilt angle.

| Parameter | Type | Description |
|-----------|------|-------------|
| `shape` | `(Lx, Ly)` | Number of cells along `a1`/`a2`. |
| `pbc_x`, `pbc_y` | `bool` | Periodic boundaries along each direction. Default `True`. |

**Returns** `Lattice`. `nb=4`.

---

### `plaquette`

*`tachys.lattice.lattice_database`*

```python
plaquette(shape, pbc_x=True, pbc_y=True)
```

2×2-site plaquette lattice with near-square intra-cell geometry (square Bravais
lattice, 4-site basis).

| Parameter | Type | Description |
|-----------|------|-------------|
| `shape` | `(Lx, Ly)` | Number of cells along `a1`/`a2`. |
| `pbc_x`, `pbc_y` | `bool` | Periodic boundaries along each direction. Default `True`. |

**Returns** `Lattice`. `nb=4`.

---

## Symmetries

### `SymOp`

*`tachys.lattice.lattice_symmetries`*

```python
class SymOp(name, matrix, perm)
```

A single named point-group symmetry operation. Extends `typing.NamedTuple`.

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Schoenflies-style label, e.g. `"C4"`, `"C6^2"`, `"σv(30°)"`, `"σd(90°)"`. |
| `matrix` | `np.ndarray` | `(2, 2)`. The Cartesian point-group matrix (identity for pure translations). |
| `perm` | `np.ndarray` | `(Ns,)` int. `perm[s]` = image of site `s` under the operation. |

---

### `translation_group`

*`tachys.lattice.lattice_symmetries`*

```python
translation_group(lat)
```

Full translation group of the finite cluster under periodic boundary
conditions. This is what selects a momentum sector: form the projector

$$
P_k = \frac{1}{N_t} \sum_t \text{phase}_t \, T_t
$$

from the permutations `T` returned here and the characters from
`momentum_phases`. The group is abelian, so its irreps are momenta. A
non-periodic direction contributes only the identity shift.

| Parameter | Type | Description |
|-----------|------|-------------|
| `lat` | `Lattice` | Lattice to compute the translation group for. |

**Returns** `(perms, labels)`.
`perms` is `(Nt, Ns)` int64: `perms[t, s]` is the image of site `s` under
translation `t`, i.e. the site whose cell is `(i+n2, j+n1) mod (Ly, Lx)`, same
sublattice. `labels` is `(Nt, 2)` int64: `(n1, n2)`, the cell shift along
`(a1, a2)` for each `t`.

---

### `momentum_phases`

*`tachys.lattice.lattice_symmetries`*

```python
momentum_phases(lat, m)
```

Characters of the translation group for momentum sector `m = (m1, m2)`, aligned
with the rows of `translation_group(lat)[0]`.

$$
\mathbf{k} = 2\pi\left(\frac{m_1}{L_x}, \frac{m_2}{L_y}\right), \qquad
\text{phase}_t = e^{-i\, \mathbf{k}\cdot\mathbf{R}_t}
$$

The projector $P_k = \frac{1}{N_t}\sum_t \text{phase}_t\, T_t$ selects states
with $T_t|\psi\rangle = e^{+i\mathbf{k}\cdot\mathbf{R}_t}|\psi\rangle$. Flip the
sign in the exponent (or negate `m`) for the opposite convention.

| Parameter | Type | Description |
|-----------|------|-------------|
| `lat` | `Lattice` | Lattice to compute phases for. |
| `m` | `(m1, m2)` | Momentum-sector indices. |

**Returns** `np.ndarray`, shape `(Nt,)`, complex128.

---

### `point_group`

*`tachys.lattice.lattice_symmetries`*

```python
point_group(lat, center=(0.0, 0.0), n_candidates=12)
```

Point-group symmetries of the finite cluster that fix `center`, returned as
named `SymOp`s. Rotations are named `C_n^p` (reduced to lowest terms, so a 60°
rotation in a hexagonal group is `C6`, a 120° one is `C3`, 180° is `C2`, ...);
reflections are `σv` (axis along a lattice vector) or `σd` (diagonal), tagged
with the axis angle. Whatever subgroup is compatible with the cluster is
detected automatically: `C4v` for square, `C6v` for triangular, a lower-order
subgroup for incompatible sizes.

`center` fixes the rotation center; for a group whose natural rotation center is
not a basis-0 atom (e.g. a kagome plaquette center), pass the correct `center`
or only the site-symmetry subgroup of the origin will be detected. Detection is
exact modulo the `retrieve_index` tolerance: an incompatible cluster shape (e.g.
a 6×4 triangular cluster that breaks 6-fold symmetry) correctly returns the
smaller compatible group rather than silently including a broken operation.

| Parameter | Type | Description |
|-----------|------|-------------|
| `lat` | `Lattice` | Lattice to compute the point group for. |
| `center` | `(x, y)` | Fixed point of the rotations. Default the origin (basis atom 0). |
| `n_candidates` | `int` | Angular resolution of the search grid. Default `12` (30° steps; enough for 2-/3-/4-/6-fold axes). |

**Returns** `list[SymOp]`.

---

### `singlet_symm`

*`tachys.lattice.symmetries`*

```python
singlet_symm(wf_apply)
```

Wrap a spin wavefunction to enforce global spin-flip (Z₂) symmetry by projecting
onto the even sector under $\sigma \to -\sigma$:

$$
\log\left[\psi(\sigma) + \psi(-\sigma)\right]
= f(\sigma) + \log\left[1 + e^{f(-\sigma) - f(\sigma)}\right]
$$

where $f(\sigma) = \log\psi(\sigma)$. Corresponds to a singlet-like ($S_z=0$)
symmetrization in the spin basis.

| Parameter | Type | Description |
|-----------|------|-------------|
| `wf_apply` | `callable` | `(params, lattice) → log-amplitude f(σ)`. |

**Returns** `callable`. Same signature, returning the symmetrized log-amplitude.

---

### `spin_flip_symm_f`

*`tachys.lattice.symmetries`*

```python
spin_flip_symm_f(wf_apply, p=1)
```

Project a spinful `FermionState` wavefunction onto the $p=\pm1$ sector of
$U = \exp(-i\pi S^y)$, the total-spin flip — the occupation-number counterpart
of `singlet_symm`, generalized to select either parity sector via `p`. `p=+1`
selects even total spin (contains $S{=}0$); `p=-1` selects odd (contains
$S{=}1$).

Assumes the doubled occupation-number layout (`n_up_1..n_up_Ns, n_dn_1..n_dn_Ns`)
used throughout tachys for single-band spinful fermions (`FermionState` with
`Nbands=2`), so up↔down flip is a half-roll of the occupations array. Requires
`N_up == N_dn` ($S_z=0$), since only then does flipping up↔down stay within the
same $(N_e, S_z)$ sector.

$$
U|n\rangle = (-1)^{N_{dn}(1+N_{up})}\,|\text{flip}(n)\rangle,\qquad
\log\left[\psi(n) + p\cdot U\text{-phase}(n)\cdot\psi(\text{flip}(n))\right]
$$

| Parameter | Type | Description |
|-----------|------|-------------|
| `wf_apply` | `callable` | `(params, state) → log-amplitude f(n)`, `state` a `FermionState` with `Nbands=2`. |
| `p` | `int` | Sector to project onto, `+1` or `-1`. Default `1`. |

**Returns** `callable`. Same signature, returning the symmetrized log-amplitude.

---

### `time_reversal`

*`tachys.lattice.symmetries`*

```python
time_reversal(apply_fn)
```

Wrap a wavefunction to enforce time-reversal symmetry by symmetrizing the
log-amplitude under complex conjugation ($\psi \to \psi^*$), producing a
real-valued wavefunction:

$$
\log\left[\psi(\sigma) + \psi^*(\sigma)\right] = \log\left[2\,\text{Re}\,\psi(\sigma)\right]
= \text{log\_amps} + \log\left[1 + e^{-2i\,\text{Im}(\text{log\_amps})}\right]
$$

The result's imaginary part encodes the sign of the wavefunction: `0` when
$\text{Re}\,\psi>0$, `π` when $\text{Re}\,\psi<0$.

| Parameter | Type | Description |
|-----------|------|-------------|
| `apply_fn` | `callable` | `(params, state) → complex log-amplitude`. |

**Returns** `callable`. Same signature, returning the time-reversal-symmetrized log-amplitude.

---

### `symmetrize_wf`

*`tachys.lattice.symmetries`*

```python
symmetrize_wf(wf_apply, perms, sector_chars=None)
```

Wrap a wavefunction to project onto a symmetric sector of a lattice permutation
group (translations, point group, or any `(M, W)` perm array). Species-agnostic:
goes through `tachys.lattice.state_array`'s `get_array`/`replace_array`, so the
same wrapper works for `SpinState` and `FermionState` alike.

If `state` is a `FermionState`, the fermionic-sign phase each group element picks
up on the occupation-number representation is added automatically (via
`sign_permutation`), so the caller only ever supplies the single forward `perms`
array — no separate inverse to build or pass in. The per-band inverse permutation
is derived from `perms` once at wrap time (not on every call).

Unlike the bosonic case, a fermionic output is not literally constant across a
group orbit: for a genuine sector eigenstate,
$f_{sym}(g.\sigma) = f_{sym}(\sigma) + i\pi\cdot[\text{sign}(g,\sigma)<0] - \log\chi(g)$
(trivial $\chi$ by default) — this is expected, not a bug: a fermionic
parity/momentum eigenstate genuinely transforms with a sign under the group.

| Parameter | Type | Description |
|-----------|------|-------------|
| `wf_apply` | `callable` | `(params, state, *args, **kwargs) → log-amplitude`. |
| `perms` | `(M, W) int array` | `W == get_array(state).shape[-1]` exactly. For a multi-band `FermionState`, build with `expand_perm(base_perms, state.Nbands)`. |
| `sector_chars` | `(M,) array` | Optional real array of π-multiples folded in as $e^{i\pi\chi}$ before combining. Default `None` (trivial character). |

**Returns** `callable`. Same signature, returning the symmetrized log-amplitude.

---

### `invert_perm`

*`tachys.lattice.symmetries`*

```python
invert_perm(perms)
```

Inverse of a site permutation, or a batch of them (any leading shape, last axis
`= Ns`). Every row is an honest bijection of `{0,...,Ns-1}`, so the inverse is
simply the argsort — no reference row or row-matching needed.

| Parameter | Type | Description |
|-----------|------|-------------|
| `perms` | `np.ndarray` | `(..., Ns)`. Permutation(s) to invert. |

**Returns** `np.ndarray`, same shape as `perms`.

---

### `expand_perm`

*`tachys.lattice.symmetries`*

```python
expand_perm(perm, n_bands)
```

Widen an `Ns`-wide site permutation (or a batch, shape `(..., Ns)`) to act on a
`State`'s physical array of width `n_bands*Ns`, by applying the same geometric
permutation independently inside each contiguous `Ns`-band slice
(`array[..., b*Ns:(b+1)*Ns]`). `n_bands=1` is a no-op (covers `SpinState`).

| Parameter | Type | Description |
|-----------|------|-------------|
| `perm` | `np.ndarray` | `(..., Ns)`. Site permutation(s). |
| `n_bands` | `int` | Number of bands to replicate the permutation across. |

**Returns** `np.ndarray`, shape `(..., n_bands*Ns)`.

---

### `combine_perm_groups`

*`tachys.lattice.symmetries`*

```python
combine_perm_groups(perms_a, perms_b, chars_a=None, chars_b=None)
```

Outer-product combine of two site-permutation groups (and, optionally, their
sector characters) into the single `(M1*M2, W)` perms / `(M1*M2,)` chars that one
`symmetrize_wf` call needs to reproduce nesting
`symmetrize_wf(symmetrize_wf(f, perms_a, chars_a), perms_b, chars_b)` exactly —
a flat `jax.lax.map` instead of `M2` sequential calls of an `M1`-step map each.

`perms_a`/`chars_a` is the group applied by the inner `symmetrize_wf` call (e.g.
translation coset reps), `perms_b`/`chars_b` the outer one (e.g. point group).
Row `(a, b)` of the combined perms is `perms_b[b][perms_a[a]]`. Entry `(a, b)` of
the combined chars is `chars_a[a] + chars_b[b]`. A missing `chars_a`/`chars_b` is
treated as all-zero (trivial character).

| Parameter | Type | Description |
|-----------|------|-------------|
| `perms_a` | `(M1, W) array` | Inner group's permutations. |
| `perms_b` | `(M2, W) array` | Outer group's permutations. |
| `chars_a` | `(M1,) array` | Optional inner sector characters. Default `None`. |
| `chars_b` | `(M2,) array` | Optional outer sector characters. Default `None`. |

**Returns** `(combined_perms, combined_chars)`. `combined_perms` has shape
`(M1*M2, W)`; `combined_chars` is `None` iff both `chars_a` and `chars_b` are `None`.

---

### `fermionic_sign`

*`tachys.lattice.symmetries`*

```python
fermionic_sign(perm)
```

Sign of `perm` via inversion counting: $(-1)^{\#\{i<j:\,\text{perm}[i]>\text{perm}[j]\}}$.
`perm` may contain the `Ns+1` sentinel in trailing slots (see `sign_permutation`)
— sentinel-vs-sentinel and sentinel-vs-real pairs never register as inversions
since the sentinel exceeds every real value, so the padding is inert. Single
sample; `vmap` at the call site for a batch.

| Parameter | Type | Description |
|-----------|------|-------------|
| `perm` | `jax.Array` | `(n,)`. Permutation (possibly sentinel-padded). |

**Returns** `int`, `+1` or `-1`.

---

### `sign_permutation`

*`tachys.lattice.symmetries`*

```python
sign_permutation(config, perm_inv)
```

Fermionic sign for one configuration (no batch axis — `vmap` this over the
MC-batch axis at the call site). `Ns` and `n_bands` are inferred from shapes:
`Ns = perm_inv.shape[-1]`, `n_bands = config.shape[-1] // Ns`. For each
`Ns`-wide band slice, finds the occupied sites (ascending, padded to length `Ns`
with sentinel `Ns+1`), maps them through `perm_inv`, and takes `fermionic_sign`;
the total sign is the product over bands.

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `jax.Array` | `(n_bands*Ns,)`. Single occupation-number configuration. |
| `perm_inv` | `jax.Array` | `(Ns,)`. Inverse site permutation. |

**Returns** `int`, `+1` or `-1`.

---

## Sign rules

Log-phase helpers for baking a fixed sign structure (Marshall sign rule, 120° classical order, …)
into a wave function's log-amplitude, so the variational ansatz only has to learn the remaining
sign-free amplitude.

### `triangular_classical_log_phase`

*`tachys.lattice.spins.sign_rules`*

```python
triangular_classical_log_phase(spins, L)
```

120°/three-sublattice classical sign rule for the triangular lattice, expressed as a log-phase.
Sublattice assignment is `(i - j) mod 3` (not `(i + j) mod 3`), matching the triangular lattice's
three nearest-neighbour bond directions a1 = (1,0), a2 = (0,1), and a1−a2 = (1,−1) (60° a1/a2
convention, see `lattice_database.triangular`); `(i + j) mod 3` is invariant along a1−a2 and is
not a valid tripartition for this bond convention. Only down spins contribute (up spins give a
factor 1, i.e. log-phase 0).

$$
\log\phi(\{S_i\}) = i\,\frac{2\pi}{3} \sum_{i:\,S_i=\downarrow} c_i, \qquad c_i = (i_\text{row} - j_\text{col}) \bmod 3
$$

| Parameter | Type | Description |
|-----------|------|-------------|
| `spins` | `jax.Array` | Shape `(batch, Lx·Ly)`. Values +1 (up) / −1 (down). Flattened with site index = `i·Lx + j` (`i` along `a2`, the row; `j` along `a1`, the column) — tachys's `Lattice._build_sites` convention. This is the transpose of `x·Ly + y` except when `Lx == Ly`. |
| `L` | `int` or `(Lx, Ly)` | Linear size. An `int` is treated as a square cluster. |

**Returns** `jax.Array`, complex, shape `(batch,)`. The full log-phase `iθ` to add to a log-amplitude.

---

### `MSR_log_phase_square`

*`tachys.lattice.spins.sign_rules`*

```python
MSR_log_phase_square(spins, L)
```

Marshall sign rule for a bipartite square lattice, as a log-phase: `log((-1)^{N_down^A})`.
The A-sublattice is the checkerboard set of sites with `(x + y) % 2 == 0`.

$$
\log\phi(\{S_i\}) = i\pi \left( N_\downarrow^A \bmod 2 \right)
$$

| Parameter | Type | Description |
|-----------|------|-------------|
| `spins` | `jax.Array` | Shape `(batch, Lx·Ly)`. Values +1 (up) / −1 (down). Row-major flattened, site index = `x·Ly + y`. |
| `L` | `int` or `(Lx, Ly)` | Linear size. An `int` is treated as a square cluster. |

**Returns** `jax.Array`, complex, shape `(batch,)`.

---

### `MSR_log_phase_chain`

*`tachys.lattice.spins.sign_rules`*

```python
MSR_log_phase_chain(spins, L)
```

Marshall sign rule for a 1-D chain, as a log-phase: `log((-1)^{N_down^A})`. The A-sublattice is
the set of even sites, `x % 2 == 0`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `spins` | `jax.Array` | Shape `(batch, L)`. Values +1 (up) / −1 (down). Site index = `x`. |
| `L` | `int` | Chain length. |

**Returns** `jax.Array`, complex, shape `(batch,)`.

---

### `add_sign_rule`

*`tachys.lattice.spins.sign_rules`*

```python
add_sign_rule(sign_fn, apply_fn, L)
```

Factory that wraps a wave function's `apply_fn` so its log-amplitude gets an additive log-phase
from `sign_fn`. Useful for combining a learned, sign-free amplitude network with a fixed,
analytically-known sign structure (e.g. `MSR_log_phase_square` or
`triangular_classical_log_phase`).

| Parameter | Type | Description |
|-----------|------|-------------|
| `sign_fn` | `callable` | `sign_fn(spins, L) -> jax.Array`. Must return the *full* log-phase (i.e. `i·θ`), shape `(batch,)`. |
| `apply_fn` | `callable` | `apply_fn(params, state, *args, **kwargs) -> jax.Array`. Returns log-amplitudes. |
| `L` | `int` or `(Lx, Ly)` | Forwarded to `sign_fn`. |

**Returns** `callable` with signature `wrapped(params, state, *args, **kwargs)`, returning
`apply_fn(params, state, *args, **kwargs) + sign_fn(state.spins, L)`.

---

## Monte Carlo

### `_BaseAction`

*`tachys.montecarlo`*

```python
class _BaseAction()
```

Abstract base for MCMC move proposals. Extends `flax.struct.PyTreeNode`. Subclass
it and implement `__call__(key, state)`. `__init_subclass__` automatically wraps
any subclass `__call__` so it always returns 4 values: implementations may return
either 3 values (atomic actions, `action_id` defaults to `0`) or 4 (when a custom
`action_id` is needed), and `log_prob_correction` is broadcast to the shape of
`allowed_move` so every action exposes a uniform output shape (required by
`jax.lax.switch` inside `CompositeAction`).

| Member | Type | Description |
|--------|------|-------------|
| `__call__(key, state)` | `(State → (State, bool[N_mc], float[N_mc], int))` | Proposes a move. Returns `(new_state, allowed_move, log_prob_correction, action_id)`. `allowed_move` is `False` for no-op moves (e.g. exchanging identical spins), allowing early rejection before the wavefunction is evaluated. `log_prob_correction` is the log-probability correction for asymmetric proposals (`0.0` for symmetric moves). `action_id` identifies which sub-action was used (scalar `0` for atomic actions; per-chain array for `CompositeAction`). |
| `n_actions` | `int` (property) | Number of distinct sub-actions. `1` for atomic actions. |

Concrete subclasses (spin-flip, bond-exchange, fermion-hop actions, …) live
alongside their respective lattice modules — see below.

---

### `CompositeAction`

*`tachys.montecarlo`*

```python
class CompositeAction(actions, probs)
```

An `_BaseAction` that randomly selects among several sub-actions independently
on each Markov chain: chain `i` applies `actions[k]` with probability `probs[k]`.

| Field | Type | Description |
|-------|------|-------------|
| `actions` | `tuple[_BaseAction, ...]` | Candidate actions to choose from. |
| `probs` | `tuple[float, ...]` | Selection probability per action. Must sum to `1` (enforced in `__post_init__`). Static (non-pytree) field. |

`n_actions` returns `len(probs)`. Calling the instance dispatches each chain to
its selected action via `jax.lax.switch` and returns the same 4-tuple as
`_BaseAction.__call__`, with `action_id` giving the per-chain index of the
sub-action actually used (useful for tracking acceptance rates per move type).

---

### `mc_step`

*`tachys.montecarlo`*

```python
mc_step(state, key, action, wf, log_amps, optimize_mask=True, batch_expand=0.25)
```

Perform one Metropolis–Hastings step across all chains: propose a move with
`action`, evaluate the wavefunction on the proposal, and accept/reject each
chain independently according to

$$
\log p_{\text{accept}} = 2\,\mathrm{Re}\big[\log\psi(s') - \log\psi(s)\big] + \Delta_{\text{corr}}
$$

where $\Delta_{\text{corr}}$ is the proposal's `log_prob_correction`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `state` | `State` | Batch axis 0 of size `N_mc`. |
| `key` | `jax.random.key[N_mc]` | One PRNG key per chain. |
| `action` | `_BaseAction` | Move-proposal callable. |
| `wf` | `WaveFunction` | Supplies `wf.apply_fn(params, state) -> (N_mc,)` log-amplitudes. |
| `log_amps` | `jax.Array`, shape `(N_mc,)` | Current log-amplitudes. |
| `optimize_mask` | `bool` | If `True`, skip wavefunction evaluation for trivially rejected moves (via `_apply_masked`). Default `True`. |
| `batch_expand` | `float` | Batch enlargement factor passed through to `_apply_masked`. Default `0.25`. |

**Returns** `(state, key, log_amps, accepted, action_id)`. `accepted` is a
boolean array of shape `(N_mc,)`; `action_id` is a scalar or per-chain integer
array identifying the sub-action used.

---

### `sample`

*`tachys.montecarlo`*

```python
sample(nsweeps, state, action, key, wf)
```

Run `nsweeps * Ns` Metropolis steps across all sharded Markov chains. JIT-compiled
and wrapped in `shard_map` over the device mesh (`tachys.parallel.mesh`, axis
`'i'`): `state` and `key` are sharded along the chain axis, `action` and `wf` are
replicated. This is the top-level entry point used by the training loop
(`tachys.ground_state_training.train`) and by `compute_observables` to advance
the Markov chain between measurements.

| Parameter | Type | Description |
|-----------|------|-------------|
| `nsweeps` | `int` | Number of sweeps; one sweep is `Ns` Metropolis steps. |
| `state` | `State` | Batch axis 0 of size `N_mc` (sharded across devices). |
| `action` | `_BaseAction` | Move-proposal callable. |
| `key` | `jax.random.key[N_mc]` | One PRNG key per chain (sharded). |
| `wf` | `WaveFunction` | The guiding wavefunction. |

**Returns** `(state, log_amps, acceptance)`. `acceptance` has shape
`(n_actions,)`: the per-action acceptance rate (accepted / selected), reduced
across all devices.

---

### Moves

Concrete `_BaseAction` subclasses used to build a `sample`/`train` call's `action`.

#### `exchange_spins`

*`tachys.lattice.spins.spin_action`*

```python
exchange_spins(spins, id1, id2)
```

Swap the values at positions `id1` and `id2` in a single (non-batched) spin or occupation array.
Used as a low-level building block by both spin and fermion move proposals.

| Parameter | Type | Description |
|-----------|------|-------------|
| `spins` | `jax.Array` | 1-D array (a single chain's spins or occupations). |
| `id1` | `int` | First index. |
| `id2` | `int` | Second index. |

**Returns** `jax.Array`, same shape as `spins`, with the two entries swapped.

---

#### `SpinFlip`

*`tachys.lattice.spins.spin_action`*

```python
class SpinFlip()
```

Proposes flipping a single, uniformly-drawn spin. Because the site is drawn uniformly, the
proposal is symmetric and `log_prob_correction = 0`.

| Member | Type | Description |
|--------|------|-------------|
| `__call__(key, state)` | `(PRNGKey, SpinState) → (SpinState, mask, 0.0)` | Flips one random site per chain. `allowed_move` is always `True`. |

---

#### `BondFlip`

*`tachys.lattice.spins.spin_action`*

```python
class BondFlip(bonds)
```

Proposes flipping **both** spins on a randomly chosen bond, unconditionally — mirroring the
process an off-diagonal bond term built from two unconditional single-site flip operators (e.g.
`Sx`/`Sy`) connects to. Unlike a swap (`BondExchange`), the flip does not require the two spins to
differ, and unlike `SpinFlip` it acts on a whole bond rather than a single site. The bond is drawn
uniformly from a fixed, precomputed list independent of the current configuration, so the
proposal is its own inverse and `log_prob_correction = 0`.

| Field / Member | Type | Description |
|-----------------|------|-------------|
| `bonds` | `tuple` | `(N_bonds, 2)` candidate `(site_i, site_j)` pairs. Static (non-pytree) field. |
| `__call__(key, state)` | `(PRNGKey, SpinState) → (SpinState, mask, 0.0)` | Flips both endpoints of a uniformly-drawn bond. `allowed_move` is always `True`. |

##### `BondFlip.create`

```python
BondFlip.create(lattice, deltas, b_from=0, b_to=None)
```

Pools one or more cell displacements into a single candidate bond list. Pass multiple `deltas`
(e.g. a Kitaev model's x- and y-bond displacements) to pool several bond types into one action.

| Parameter | Type | Description |
|-----------|------|-------------|
| `lattice` | `Lattice` | Lattice providing bond geometry via `lattice.bonds`. |
| `deltas` | `iterable of (int, int)` | Cell displacements, each passed to `lattice.bonds`. |
| `b_from` | `int` | Source sublattice. Default `0`. |
| `b_to` | `int` or `None` | Target sublattice. Default: same as `b_from`. |

**Returns** `BondFlip`.

---

#### `BondExchange`

*`tachys.lattice.bond_exchange`*

```python
class BondExchange(max_dist=1, bonds, Nbands=1)
```

Extends `tachys.montecarlo._BaseAction`. Proposes exchanging two sites' values
within the same band.

Candidate site pairs are precomputed from the lattice geometry (all bonds up to
`max_dist` shells). On each step, a band is drawn uniformly, then a pair is drawn
uniformly among the bonds (within that band) that are currently valid (the two
sites differ). Because the proposal is restricted to the valid subset, and the
count of valid bonds generally differs before/after the move, the proposal is
asymmetric and needs a log-probability correction.

Works with any `State` subclass supported by `tachys.lattice.state_array`'s
`get_array`/`replace_array` (`SpinState.spins` or `FermionState.occupations`).

| Field | Type | Description |
|-------|------|-------------|
| `max_dist` | `int` | Maximum bond distance (in lattice shells) between the two sites. Static field. |
| `bonds` | `tuple` | Candidate `(site_i, site_j)` pairs, as a hashable tuple-of-tuples (not a `jnp.array`) so instances stay hashable for `jax.lax.switch` inside `CompositeAction`. Static field. |
| `Nbands` | `int` | Number of bands sharing the same site indexing (e.g. `2` for spin-½ fermion occupations, `1` for plain spins). Default `1`. Static field. |

##### `BondExchange.create`

```python
BondExchange.create(lattice, max_dist=1, Nbands=1)
```

Classmethod constructor. Builds the candidate bond list from
`lattice.shells(max_dist)`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `lattice` | `Lattice` | Lattice geometry to build candidate bonds from. |
| `max_dist` | `int` | Maximum bond distance (in shells) to include. Default `1`. |
| `Nbands` | `int` | Number of bands sharing the site indexing. Default `1`. |

**Returns** `BondExchange`.

##### `BondExchange.__call__`

```python
bond_exchange(key, state)
```

Propose one bond-exchange move per walker: draw a band uniformly, then a bond
uniformly among the currently-valid bonds (the two sites' values differ) in that
band, and swap the two sites' values.

| Parameter | Type | Description |
|-----------|------|-------------|
| `key` | `PRNGKey` | Batched JAX random key, shape `(N_mc_local, 2)`. |
| `state` | `State` | Current walker batch. |

**Returns** `(new_state, allowed_move, log_prob_correction)`. `new_state` is
`state` with the two chosen sites' values exchanged; `allowed_move` is always
`True` by construction; `log_prob_correction` is
`log(N_valid_before) - log(N_valid_after)`, the asymmetric-proposal correction.

---

#### `FermionSpinExchange`

*`tachys.lattice.fermions.fermion_action`*

```python
class FermionSpinExchange(max_dist=1, bonds=...)
```

Proposes exchanging the local spin between two singly-occupied sites within `max_dist` lattice
neighbour shells of each other: an up electron at site `i` becomes a down electron at the same
site `i`, and a down electron at a neighbouring site `j` becomes an up electron at that same site
`j` (`i` and `j` are drawn, in random order, from a precomputed candidate bond list). No electron
actually hops — each flips its own band in place — but the net effect on sites `i` and `j` is the
same as swapping their (opposite) spin orientations.

Combined, the move conserves `Nup` and `Ndn`. It requires site `i` to currently hold an up
electron with its down slot empty, and site `j` to hold a down electron with its up slot empty;
`allowed_move` is `False` otherwise (occupied-source / empty-destination guard).

The bond is drawn uniformly from the fixed, precomputed `bonds` list (not by picking a site and
then one of its neighbours), so the proposal stays symmetric and `log_prob_correction = 0` even
under OBC, where boundary sites have fewer neighbours than bulk sites (a site-then-neighbour
scheme would implicitly weight by `1/degree(site)` and break detailed balance there). Unlike
`BondExchange`, the random choice is not restricted to only currently-valid bonds — this move's
validity condition (occupied + empty on both sides of the bond) is stronger than
`BondExchange`'s, so a walker could plausibly have zero valid bonds at some step; proposing
uniformly and rejecting invalid draws via `allowed_move` sidesteps that.

| Field / Member | Type | Description |
|-----------------|------|-------------|
| `max_dist` | `int` | Maximum bond distance (in lattice shells) between the two sites. Static (non-pytree) field. |
| `bonds` | `tuple` | `(N_bonds, 2)` candidate `(site_i, site_j)` pairs, precomputed from the lattice geometry up to `max_dist` shells. Static (non-pytree) field. |
| `__call__(key, state)` | `(PRNGKey, FermionState) → (FermionState, mask, 0.0)` | Draws a bond and, in a random order, moves an up electron and a down electron as described above. |

##### `FermionSpinExchange.create`

```python
FermionSpinExchange.create(lattice, max_dist=1)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `lattice` | `Lattice` | Lattice object providing neighbour-shell geometry via `lattice.shells`. |
| `max_dist` | `int` | Number of neighbour shells (by distance) to draw candidate bonds from. Default `1`. |

**Returns** `FermionSpinExchange`.

---

## Wavefunction ansätze

All classes below are `flax.linen.Module` subclasses representing variational wavefunctions for Monte Carlo sampling. They follow the standard Flax lifecycle and a shared calling convention used throughout `tachys`:

- **Init**: `params = model.init(key, state)`, where `state` is a representative `SpinState`, `FermionState`, or a foundation-model variant carrying an extra `system_couplings` field.
- **Apply**: `log_psi = model.apply(params, state)` evaluates the log-wavefunction on a batch of configurations (leading batch axis); the result has shape `(batch,)`. Modules only ever receive batches: when tachys evaluates a single configuration (the per-sample Jacobians of the optimizers), `WaveFunction` adds the batch axis (see below). `model.init` is not wrapped, so call it with a batch.
- **Complex vs. real output**: whenever the architecture derives its output from `jnp.linalg.slogdet` (all fermionic/determinant ansätze), or is explicitly constructed with `complex=True` (RBM/ViT ansätze), `log_psi` is complex: `Re[log_psi] = log|ψ|` is the log-amplitude and `Im[log_psi]` is the phase, so `ψ = exp(log_psi)`. With `complex=False`, RBM-style ansätze return a real log-amplitude only (a sign/phase-free wavefunction).
- To drive sampling/optimization (`tachys.montecarlo`, `tachys.optimizer`), wrap `(params, model.apply)` in a `tachys.wavefunction.WaveFunction`.

### Wavefunction container

#### `WaveFunction`

*`tachys.wavefunction`*

```python
class WaveFunction(params, apply_fn, unravel_params_fn=None, dtype=jnp.float64)
```

Immutable container pairing a parameter pytree with its apply function. Extends `flax.struct.PyTreeNode`. This is the object passed to `tachys.montecarlo.sample`, `tachys.lattice.operator.local_estimator`, and the optimizers in `tachys.optimizer` — none of them call a model directly, they all go through `wf.apply_fn(wf.params, state)`.

| Member | Type | Description |
|--------|------|-------------|
| `params` | pytree | Model parameters (a pytree node, tracked by JAX transformations). |
| `apply_fn` | `Callable` | Static (non-pytree) field. Typically `model.apply` for some `nn.Module`. Called as `apply_fn(params, state) -> log_psi`. Wrapped in `__post_init__` so that the function always receives a batch: a single configuration (no batch axis) gets a leading axis of size one on every data leaf and returns a single log-amplitude. The wrapping is done once; the original function is `wf.apply_fn.__wrapped__`. |
| `unravel_params_fn` | `Callable` | Static field. Maps a flat parameter vector back to the `params` pytree structure. If not supplied, computed automatically in `__post_init__` via `jax.flatten_util.ravel_pytree(params)`. |
| `dtype` | `Any` | Static field. Default `jnp.float64`. |
| `apply_gradients(grads, eta)` | `(pytree, float) → WaveFunction` | `jax.jit`-compiled plain gradient-descent step: `new_params = params - eta * grads`, returned as a new `WaveFunction` via `.replace(...)`. |
| `num_params` | `int` (property) | Total number of scalar parameters, computed as the flattened size of `params` via `ravel_pytree`. |

---

### Restricted Boltzmann machine (RBM) ansätze

Single-hidden-layer RBM wavefunctions. `SpinRBM` acts directly on spin configurations; `FermionRBM` uses a neural backflow correction on top of a bare Slater determinant.

#### `log_cosh`

*`tachys.lattice.ansatz.rbm`*

```python
log_cosh(x)
```

Numerically stable elementwise `log(cosh(x))`, implemented as `|x| + log1p(exp(-2|x|)) - log(2)` (sign of `x.real` tracked separately so it also works for complex `x`). Used as the nonlinearity in every RBM/output-head module in this package.

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `jax.Array` | Real or complex array. |

**Returns** `jax.Array`, same shape as `x`.

---

#### `SpinRBM`

*`tachys.lattice.ansatz.rbm`*

```python
class SpinRBM(hidden_units, dtype=jnp.float64, complex=False)
```

Complex restricted Boltzmann machine over spin-½ configurations. A single dense (visible→hidden) layer produces a pre-activation per hidden unit; when `complex=True` a second, independently-parametrized dense layer supplies the imaginary part of that pre-activation (there is no single complex-valued kernel — real and imaginary parts come from two real `nn.Dense` layers). The log-wavefunction is the sum over hidden units of `log_cosh` of the (possibly complex) pre-activation:

$$
\log\psi(s) = \sum_{k=1}^{H} \log\cosh\!\big(z_k(s)\big), \qquad
z_k(s) = \sum_i W_{ki} s_i + b_k \;+\; i\Big(\sum_i W'_{ki} s_i + b'_k\Big)\ \text{(if complex)}
$$

| Field | Type | Description |
|-------|------|-------------|
| `hidden_units` | `int` | Number of hidden units `H`. |
| `dtype` | `Any` | Parameter dtype for both dense layers. Default `jnp.float64`. |
| `complex` | `bool` | If `True`, adds a second dense layer (`imag_linear`) whose output becomes the imaginary part of the pre-activation, making `log_psi` complex (amplitude + phase). If `False`, `log_psi` is real. Default `False`. |

**Call signature** `__call__(lattice) -> jax.Array`, shape `(batch,)`. `lattice` is a `SpinState`; only `lattice.spins` (values in {−1, +1}) is used.

---

#### `FermionRBM`

*`tachys.lattice.ansatz.rbm`*

```python
class FermionRBM(hidden_units)
```

Backflow-corrected Slater-determinant ansatz for spinful/multiband fermions. A bare set of `Ne` orbitals over all fermionic modes is held as a direct parameter; a two-layer `tanh` MLP ("backflow network") maps the *full* occupation-number vector of a sample to a per-sample additive correction to those orbitals. The wavefunction is the determinant of the corrected orbital matrix evaluated at the occupied positions:

$$
\psi(n) = \det\Big[\, \phi_{a}(r_i) + F_{a}(n)_{i} \,\Big]_{a,i=1}^{N_e}, \qquad r_1,\dots,r_{N_e} = \text{occupied modes of } n
$$

| Field | Type | Description |
|-------|------|-------------|
| `hidden_units` | `int` | Width of the backflow MLP's hidden layer. |

Internally, `orbitals` is a `(lattice.Ne, N_modes)` parameter (`nn.initializers.xavier_uniform`, dtype `float64`), where `N_modes = occupations.shape[-1]` is the *total* number of fermionic modes (e.g. `lattice.Ns * lattice.Nbands` for a spinful system — not the single-band site count). The backflow MLP is `Dense(hidden_units) → tanh → Dense(N_modes * Ne)`, reshaped to `(batch, Ne, N_modes)` and added to the bare orbitals. For each sample, the `Ne` occupied mode indices `R` (via `.nonzero(size=Ne)`) select the corresponding columns, giving an `(Ne, Ne)` matrix passed to `_log_det` (see below).

**Call signature** `__call__(lattice) -> jax.Array`, shape `(batch,)`, always complex (result of `_log_det`). `lattice` is a `FermionState`; uses `lattice.occupations` and `lattice.Ne`.

**Returns** log-amplitude + phase; `-inf` (real part) for samples where the orbital matrix is singular.

---

### Foundation-model RBM ansätze

*`tachys.lattice.ansatz.rbm_foundation`*

Drop-in generalizations of `SpinRBM`/`FermionRBM` for training one shared network across *multiple* Hamiltonians simultaneously. Each sample's Hamiltonian coupling vector (`lattice.system_couplings`, produced by `tachys.lattice.foundation.operators.extract_system_couplings`) is concatenated to the network's input so a single set of weights can condition its output on which system the sample came from.

#### `SpinFoundationRBM`

```python
class SpinFoundationRBM(hidden_units, dtype=jnp.float64, complex=False)
```

Identical architecture and fields to `SpinRBM`, except the dense layer's input is `concatenate([lattice.spins, lattice.system_couplings], axis=-1)` rather than `lattice.spins` alone.

| Field | Type | Description |
|-------|------|-------------|
| `hidden_units` | `int` | Number of hidden units. |
| `dtype` | `Any` | Parameter dtype. Default `jnp.float64`. |
| `complex` | `bool` | Same meaning as in `SpinRBM`. Default `False`. |

**Call signature** `__call__(lattice) -> jax.Array`, shape `(batch,)`. `lattice` must additionally provide `system_couplings`, shape `(batch, N_couplings)`.

---

#### `FermionFoundationRBM`

```python
class FermionFoundationRBM(hidden_units)
```

Foundation-model generalization of `FermionRBM`. The bare Slater-determinant `orbitals` parameter stays system-independent (shared across all Hamiltonians); only the backflow correction network is conditioned on the couplings — its input is `concatenate([occupations, system_couplings], axis=-1)`.

| Field | Type | Description |
|-------|------|-------------|
| `hidden_units` | `int` | Width of the backflow MLP's hidden layer. |

**Call signature** `__call__(lattice) -> jax.Array`, shape `(batch,)`, complex. `lattice` is a foundation `FermionState` exposing `occupations`, `Ne`, and `system_couplings`.

---

### Transformer building blocks

*`tachys.lattice.ansatz.transformer.attention`*, *`tachys.lattice.ansatz.transformer.encoder`*

Shared building blocks used by both `FermionicTransformer` and `SpinViT`. The attention mechanism is unusual: instead of learning query/key projections, attention weights are learned directly as a parameter (optionally forced to be translation-invariant), and only a value projection is data-dependent.

#### `FactoredAttention`

*`tachys.lattice.ansatz.transformer.attention`*

```python
class FactoredAttention(d_model, num_heads, seq_len, dtype, transl_invariant=False, two_dimensional=False)
```

Multi-head attention where the attention-weight matrix `alpha` (per head) is a free parameter rather than a function of the input — there is no query/key projection, only a value projection `v = Dense(x)` and an output projection `W`. The output for head `h` is `alpha_h @ v_h`.

When `transl_invariant=True`, only a single length-`seq_len` row per head is learned and every other row is obtained by `jnp.roll`-ing it, producing a circulant (translation-invariant) attention matrix rather than a full unconstrained `(seq_len, seq_len)` matrix. When `two_dimensional=True` in addition, the sequence is interpreted as a flattened `√seq_len × √seq_len` square lattice, and the roll is applied independently along both spatial axes (`roll2d`), enforcing 2D translational invariance; this requires `seq_len` to be a perfect square and `transl_invariant=True`.

| Field | Type | Description |
|-------|------|-------------|
| `d_model` | `int` | Total embedding dimension; must be divisible by `num_heads`. |
| `num_heads` | `int` | Number of attention heads. |
| `seq_len` | `int` | Sequence length. |
| `dtype` | `Any` | Parameter dtype. |
| `transl_invariant` | `bool` | Enforce translational invariance via rolling a single learned row. Default `False`. |
| `two_dimensional` | `bool` | Enforce 2D translational invariance (requires `transl_invariant=True` and `seq_len` a perfect square). Default `False`. |

**Call signature** `__call__(x) -> jax.Array`, `x` shape `(batch, seq_len, d_model)` → output same shape.

---

#### `EncoderBlock`

*`tachys.lattice.ansatz.transformer.encoder`*

```python
class EncoderBlock(d_model, num_heads, seq_len, dtype, transl_invariant=False, two_dimensional=False)
```

One standard pre-LayerNorm transformer block: `x = x + FactoredAttention(LayerNorm(x))`, then `x = x + FFN(LayerNorm(x))`, where the feed-forward network is `Dense(4·d_model) → gelu → Dense(d_model)`.

| Field | Type | Description |
|-------|------|-------------|
| `d_model` | `int` | Embedding dimension. |
| `num_heads` | `int` | Number of attention heads, forwarded to `FactoredAttention`. |
| `seq_len` | `int` | Sequence length, forwarded to `FactoredAttention`. |
| `dtype` | `Any` | Parameter dtype. |
| `transl_invariant` | `bool` | Forwarded to `FactoredAttention`. Default `False`. |
| `two_dimensional` | `bool` | Forwarded to `FactoredAttention`. Default `False`. |

**Call signature** `__call__(x) -> jax.Array`, shape `(batch, seq_len, d_model)` → same shape.

---

#### `Encoder`

*`tachys.lattice.ansatz.transformer.encoder`*

```python
class Encoder(num_layers, d_model, num_heads, seq_len, dtype, transl_invariant=False, two_dimensional=False)
```

Stack of `num_layers` `EncoderBlock`s applied sequentially (no positional embedding is added — translational structure, if any, comes entirely from `FactoredAttention`'s `transl_invariant`/`two_dimensional` options).

| Field | Type | Description |
|-------|------|-------------|
| `num_layers` | `int` | Number of stacked `EncoderBlock`s. |
| `d_model` | `int` | Embedding dimension. |
| `num_heads` | `int` | Attention heads per block. |
| `seq_len` | `int` | Sequence length. |
| `dtype` | `Any` | Parameter dtype. |
| `transl_invariant` | `bool` | Forwarded to every block. Default `False`. |
| `two_dimensional` | `bool` | Forwarded to every block. Default `False`. |

**Call signature** `__call__(x) -> jax.Array`, shape `(batch, seq_len, d_model)` → same shape.

---

### Fermionic transformer ansatz

*`tachys.lattice.ansatz.fermionic_transformer`*

A transformer-encoder backflow producing a single Slater determinant over fermionic modes, for spinful (two-band) fermion configurations.

#### `_log_det`

```python
_log_det(A)
```

Numerically robust `log(det(A))` for a batch of square matrices, returned as a complex number: `Re = log|det A|`, `Im = arg(det A)` (i.e. `0` for positive real determinant, `π` for negative). Computed via `jnp.linalg.slogdet`; the result dtype is promoted to at least `complex64`. Any `NaN` (e.g. from a singular matrix) is replaced with `-inf`. Shared by `FermionRBM`, `OutputHeadDet`, and hence `FermionicTransformer`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `A` | `jax.Array` | Shape `(..., N, N)`. |

**Returns** `jax.Array`, shape `(...)`, complex dtype.

---

#### `compute_orbitals_fn`

```python
compute_orbitals_fn(y, weights)
```

Contracts per-mode transformer features against a per-mode orbital-weight tensor to produce `Ne` orbital values per mode: `einsum('batch Norb d, Norb d Ne -> batch Norb Ne')`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `y` | `jax.Array` | Shape `(batch, N_orb, d_model)`. Per-mode feature vectors (e.g. transformer encoder output). |
| `weights` | `jax.Array` | Shape `(N_orb, d_model, Ne)`. Per-mode orbital projection weights. |

**Returns** `jax.Array`, shape `(batch, N_orb, Ne)`.

---

#### `OutputHeadDet`

```python
class OutputHeadDet(d_model, Ne, Ns, dtype, Nbands=2)
```

Slater-determinant output head. Duplicates the per-site encoder output `y` (shape `(batch, Ns, d_model)`) into two copies concatenated along the mode axis (`(batch, 2·Ns, d_model)`) — one range of the learned weight tensor's leading axis effectively serves each spin band — projects each of the `2·Ns` modes to `Ne` orbital values via `compute_orbitals_fn`, gathers the rows at the `Ne` occupied mode positions `R`, and returns `_log_det` of the resulting `(Ne, Ne)` matrix.

| Field | Type | Description |
|-------|------|-------------|
| `d_model` | `int` | Encoder feature dimension. |
| `Ne` | `int` | Number of electrons (determinant size). |
| `Ns` | `int` | Number of lattice sites (the mode tensor spans `2·Ns` = both bands). |
| `dtype` | `Any` | Parameter dtype for the orbital weight tensor `W`, shape `(2·Ns, d_model, Ne)`. |
| `Nbands` | `int` | Accepted but **not used** in `setup`/`__call__` — the orbital tensor's leading dimension is hard-coded to `2·Ns` regardless of this value. Default `2`. |

**Call signature** `__call__(y, R) -> jax.Array`. `y` shape `(batch, Ns, d_model)`; `R` shape `(batch, Ne)`, integer indices of occupied modes in `[0, 2·Ns)`.

**Returns** `jax.Array`, shape `(batch,)`, complex (log-amplitude + phase).

---

#### `FermionicTransformer`

```python
class FermionicTransformer(num_layers, d_model, num_heads, Ne, Ns, Nbands=2, dtype=jnp.float64, transl_invariant=True, two_dimensional=True)
```

Transformer-backflow Slater-determinant wavefunction for two-band (spin-↑/↓) fermions on a lattice. Each site's local occupation (an integer in `[0, 2**Nbands)` combining its up/down occupation bits) is embedded with `nn.Embed`, run through a translation-invariant `Encoder`, layer-normed, and fed to `OutputHeadDet` together with the positions of the occupied modes to produce a single determinant amplitude.

Note: although `Nbands` is a generic field, `__call__` hard-codes a two-band split (`n[..., :Ns]` = band 0, `n[..., Ns:]` = band 1) and combines them via `2**jnp.arange(Nbands)`; using `Nbands != 2` will raise a shape error. `Ns` here means the single-band lattice site count (`state.occupations` has shape `(batch, 2·Ns)`), unlike the `Ns` convention in `FermionRBM`.

| Field | Type | Description |
|-------|------|-------------|
| `num_layers` | `int` | Number of transformer encoder layers. |
| `d_model` | `int` | Transformer embedding dimension. |
| `num_heads` | `int` | Attention heads per encoder layer. |
| `Ne` | `int` | Number of electrons. |
| `Ns` | `int` | Number of lattice sites (per band). |
| `Nbands` | `int` | Number of bands; must be `2` for the current implementation. Default `2`. |
| `dtype` | `Any` | Parameter dtype. Default `jnp.float64`. |
| `transl_invariant` | `bool` | Passed to the internal `Encoder`/`FactoredAttention`. Default `True`. |
| `two_dimensional` | `bool` | Passed to the internal `Encoder`/`FactoredAttention` (requires `Ns` to be a perfect square). Default `True`. |

**Call signature** `__call__(state) -> jax.Array`, shape `(batch,)`, complex. `state` is a `FermionState`; uses `state.occupations`, `state.Ns`, `state.Ne`.

---

### Vision-transformer (ViT) spin ansatz

*`tachys.lattice.ansatz.spin_vit`*

A patch-based vision-transformer wavefunction for spin-½ configurations: the lattice is partitioned into small patches, each patch is linearly embedded, a translation-invariant transformer encoder mixes patches, and a pooled, `log_cosh`-nonlinear output head produces the (optionally complex) log-amplitude.

#### `extract_patches1d`

```python
extract_patches1d(x, b)
```

Splits a 1D chain of sites into non-overlapping patches of size `b`: `rearrange(x, 'batch (seq_len b) -> batch seq_len b', b=b)`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `jax.Array` | Shape `(batch, N)`. |
| `b` | `int` | Patch size; must divide `N`. |

**Returns** `jax.Array`, shape `(batch, N/b, b)`.

---

#### `extract_patches2d`

```python
extract_patches2d(x, b)
```

Splits a flattened square lattice of `N = L²` sites into non-overlapping `b×b` patches, and flattens each patch to a vector. Concretely, reshapes `x` to `(batch, L, b, L, b)`... after transposing and reshaping, returns `(batch, (L/b)², b²)` — `(L/b)²` patches, each a length-`b²` vector.

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `jax.Array` | Shape `(batch, N)` with `N = L²` a perfect square. |
| `b` | `int` | Patch side length; must divide `L`. |

**Returns** `jax.Array`, shape `(batch, (L/b)², b²)`.

---

#### `Embed`

```python
class Embed(d_model, b, dtype, two_dimensional=False)
```

Patch-extraction + linear embedding layer (standard ViT "patchify"). Uses `extract_patches2d` when `two_dimensional=True`, otherwise `extract_patches1d`, then applies a shared `nn.Dense(d_model)` to every patch vector.

| Field | Type | Description |
|-------|------|-------------|
| `d_model` | `int` | Output embedding dimension per patch. |
| `b` | `int` | Patch size (side length if `two_dimensional`). |
| `dtype` | `Any` | Parameter dtype. |
| `two_dimensional` | `bool` | Use 2D (square-lattice) patch extraction instead of 1D chunking. Default `False`. |

**Call signature** `__call__(x) -> jax.Array`, `x` shape `(batch, N)` → output shape `(batch, num_patches, d_model)`.

---

#### `OutputHead`

```python
class OutputHead(d_model, dtype, complex)
```

Pools the encoder's per-patch outputs by summation, layer-norms, and projects through one (or two, if complex) `Dense → LayerNorm` branches before applying the `log_cosh` nonlinearity and summing over the feature axis to produce a scalar log-amplitude — the same read-out pattern as the RBM ansätze, applied on top of transformer features instead of raw spins.

$$
z = \text{LayerNorm}\Big(\textstyle\sum_{\text{patches}} y\Big), \qquad
\text{out} = \text{LN}_2(\text{Dense}_0(z)) \;+\; i\,\text{LN}_3(\text{Dense}_1(z))\ \text{(if complex)}, \qquad
\log\psi = \sum \log\cosh(\text{out})
$$

| Field | Type | Description |
|-------|------|-------------|
| `d_model` | `int` | Feature dimension of the pooled representation and both output `Dense` layers. |
| `dtype` | `Any` | Parameter dtype. |
| `complex` | `bool` | If `True`, adds a second `Dense`+`LayerNorm` branch (`output_layer1`/`norm3`) providing the imaginary part; `log_psi` is then complex. If `False`, `log_psi` is real. |

**Call signature** `__call__(y) -> jax.Array`, `y` shape `(batch, num_patches, d_model)` → output shape `(batch,)`.

---

#### `SpinViT`

```python
class SpinViT(num_layers, d_model, num_heads, seq_len, b, complex=True, transl_invariant=False, two_dimensional=False, dtype=jnp.float64)
```

Full vision-transformer wavefunction for spin-½ configurations: `Embed` → `Encoder` (stack of `FactoredAttention`-based blocks) → `OutputHead`. The call is wrapped in `nn.remat` (gradient checkpointing) to reduce memory use during backpropagation through the encoder stack.

| Field | Type | Description |
|-------|------|-------------|
| `num_layers` | `int` | Number of transformer encoder layers. |
| `d_model` | `int` | Patch embedding / transformer dimension. |
| `num_heads` | `int` | Attention heads per encoder layer. |
| `seq_len` | `int` | Number of patches produced by `Embed`; must match `b` and the lattice size (`N/b` for 1D, `(L/b)²` for 2D), and is used to size the `FactoredAttention` weight tensors. |
| `b` | `int` | Patch size (side length if `two_dimensional`). |
| `complex` | `bool` | Forwarded to `OutputHead`; controls whether `log_psi` is complex. Default `True`. |
| `transl_invariant` | `bool` | Forwarded to `Encoder`/`FactoredAttention`. Default `False`. |
| `two_dimensional` | `bool` | Use 2D patch extraction and 2D translation-invariant attention (requires `transl_invariant=True` for the latter to take effect). Default `False`. |
| `dtype` | `Any` | Parameter dtype throughout. Default `jnp.float64`. |

**Call signature** `__call__(lattice) -> jax.Array`, shape `(batch,)`. `lattice` is a `SpinState`; uses `lattice.spins`.

**Returns** log-amplitude (complex if `complex=True`, else real).

---

## Training

### `train`

*`tachys.ground_state_training`*

```python
train(key, H, state, wf, optimizer, action, N_steps, lr_schedule, N_mc,
      wandb_run=None, log_callback_fn=None, skip_optimization=False, nsweeps=1,
      opt_state=None, start_step=0, estimator=None)
```

Run the main VMC ground-state optimization loop: at every step, sample the
Markov chain with `sample`, evaluate the local energy and its moments with
`compute_expectation`, take one optimizer step (e.g. `SR`, `SPRING`, `MARCH`),
and update `wf`'s parameters. Prints a live per-step diagnostics table (energy
per site, variance, V-score, acceptance, timings, ETA) and optionally logs to
a caller-supplied `wandb` run and checkpoints via `tachys.checkpoint`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `key` | `jax.random.key` | PRNG key. |
| `H` | `_Operator` | Hamiltonian whose expectation value is minimized. |
| `state` | `State` | Initial Monte Carlo configuration batch. `state.lattice.Ns` sets the energy-per-site normalization. |
| `wf` | `WaveFunction` | Variational wavefunction to be optimized in place (functionally — a new `wf` is returned each step). |
| `optimizer` | optimizer (`SR`, `SPRING`, `MARCH`, …) | Natural-gradient optimizer; called as `optimizer(E_L, opt_state, state, wf)`. |
| `action` | `_BaseAction` | MCMC move proposal used for sampling. |
| `N_steps` | `int` | Number of optimization steps to run in *this* call. |
| `lr_schedule` | `callable(step: int) -> float` | Learning rate as a function of the absolute step. |
| `N_mc` | `int` | Number of Markov chains. |
| `wandb_run` | optional wandb run | If given, logs `lr`, `energy`, `variance_per_site`, `vscore`, `acceptance` every step and checkpoints `wf.params`/`state`/`opt_state`/`key` into `wandb_run.dir/checkpoints` every `wandb_run.config["checkpoint_every"]` steps (default: once, at the end). Caller owns its lifecycle (`wandb.init`/`.finish`); expected non-`None` only on the `MASTER` rank — every rank still participates in the collective checkpoint calls. |
| `log_callback_fn` | optional `callable(state, wf, step) -> dict \| None`, or list thereof | Extra metrics merged into the wandb log. `None` results are skipped. Called on *every* rank regardless of `wandb_run` (callbacks are typically jitted and may touch mesh-sharded arrays, so all ranks must call them in lockstep); only the merged result is logged, and only on `MASTER`. |
| `skip_optimization` | `bool` | If `True`, skip the optimizer step / parameter update each iteration — only sample and evaluate the energy of `wf`. Default `False`. |
| `nsweeps` | `int` | MC sweeps per step, passed to `sample`. Default `1`. |
| `opt_state` | optional optimizer state | Pre-initialized optimizer state (e.g. restored via `tachys.checkpoint.load_checkpoint`) to resume from. Defaults to a fresh `optimizer.init(wf.params)`. |
| `start_step` | `int` | Absolute step number to resume at. Offsets `lr_schedule`, the wandb log step, the printed step column, and checkpoint numbering. `N_steps` still counts iterations run by *this* call — pass the remaining steps, not the original total. |
| `estimator` | optional callable | Replaces the default `\|ψ\|²` expectation value with an importance-weighted one (see [Reweighted estimators](#reweighted-estimators)). `None` (default) calls `compute_expectation` directly and leaves the loop bit-identical. |

**Returns** `(key, state, wf, opt_state, history)`. `history` is a
`dict[str, list]` with keys `"energy"`, `"variance_per_site"`, `"vscore"`,
`"acceptance"`, `"lr"`, one entry per step.

---

(reweighted-estimators)=

#### Reweighted estimators

`estimator` is the seam for sampling from a density other than `|ψ|²` and
correcting for it with per-sample importance weights. The protocol is

```python
estimator(keys, H, wf, state, log_amps)
    -> (eval_state, E_L, weights, e_mean, e2_mean, metrics)
```

| Element | Description |
|---------|-------------|
| `keys` | `(N_mc,)` per-chain PRNG keys, built as `jax.random.split(key, N_mc)` — the same form `sample` takes. |
| `eval_state` | The batch `E_L` was actually evaluated on. This — not the chain's `state` — is what `train` hands the optimizer, because the weights correct *those* configurations back to `|ψ|²`. |
| `weights` | `None`, or a `(N_mc,)` array of per-sample importance weights. The overall scale does not matter: `_BaseOptimizer._call_reweighted` divides by the psum'd mean weight before anything downstream sees them, so the update is invariant under `w → c·w`. |
| `metrics` | `dict[str, float]` of extra scalars, merged into the wandb log and appended to the printed per-step line. |

The Markov chain always carries the unmodified `state` forward, and
`log_callback_fn` and the checkpoints keep seeing it too — an estimator changes
what the energy and the gradient are computed from, never what is sampled.

`weights` activates `_BaseOptimizer._call_reweighted`, i.e. the weighted NTK
centering and `sqrt(w)` scaling in `tachys.optimizer._kernels`. That path raises
`NotImplementedError` for a `FoundationState`, so reweighted estimators do not
currently work with foundation models.

---

### `compute_observables`

*`tachys.ground_state_training`*

```python
compute_observables(key, N_steps, state, action, wf, N_mc, op_groups, nsweeps=1, log_every=1)
```

Measure a fixed set of observables along a Markov chain with `wf` held fixed —
unlike `train`, this never updates parameters. Every operator in `op_groups` is
evaluated on the same sampled batch at each step, so different observables
share Monte Carlo statistics rather than being measured from independent runs.

| Parameter | Type | Description |
|-----------|------|-------------|
| `key` | `jax.random.key` | PRNG key. |
| `N_steps` | `int` | Number of sampling steps (measurements). |
| `state` | `State` | Current Monte Carlo configuration batch. |
| `action` | `_BaseAction` | MCMC move proposal used for sampling. |
| `wf` | `WaveFunction` | Fixed guiding wavefunction. |
| `N_mc` | `int` | Number of Markov chains. |
| `op_groups` | `dict[str, Sequence[_Operator]]` or `Sequence[_Operator]` | Named groups of observables (e.g. the output of an observable-construction helper). Every operator of every group is evaluated at every step. A bare sequence is treated as a single group named `"obs"`. |
| `nsweeps` | `int` | MC sweeps per step, passed to `sample`. Default `1`. |
| `log_every` | `int` | Print a status line every this many steps. `0` disables. Default `1`. |

**Returns** `(key, state, metrics)`. `metrics` is `dict[str, np.ndarray]`;
`metrics[name]` has shape `(N_steps, len(op_groups[name]))`, the real part of
`⟨O⟩` at every step.

---

## Real-time dynamics

Real-time evolution (t-VMC) mirrors ground-state optimization: the same sampler,
the same local estimator, the same neural tangent kernel, the same VJP back to
parameter space. Three things change.

**The equation.** Minimizing the residual of the linearized evolution,
$\lVert \sum_k \dot\theta_k \lvert\partial_k\psi\rangle + i(H - \langle H\rangle)\lvert\psi\rangle\rVert^2$,
over *real* $\dot\theta$ gives

$$
S\,\dot\theta = \mathrm{Im}\,F, \qquad
S_{kl} = \mathrm{Re}\langle \Delta O_k^{*}\,\Delta O_l\rangle, \qquad
F_k = \langle \Delta O_k^{*}\,\Delta E_L\rangle,
$$

whereas imaginary time (`SR`) gives $S\dot\theta = -\mathrm{Re}\,F$, i.e. the
natural gradient $S^{-1}\nabla E$ with $\nabla E = 2\,\mathrm{Re}\,F$. So real
time is imaginary time with the generator multiplied by $i$ — in the NTK
formulation, one line: the force vector becomes
$\varepsilon_i = i\,(E_{L,i} - \bar E_L)^{*}/\sqrt{N_{mc}}$ (note both the `1j`
*and* the dropped factor of 2 relative to `SR`, which the ground-state learning
rate absorbs but a physical time step cannot).

**The regularization.** The kernel is genuinely rank deficient here — centering
alone puts exact zero modes in the spectrum, and $2N_{mc} > n_{params}$ makes it
singular by construction. `TDVP` inverts it by diagonalization and discards
eigenvalues below a threshold (`linear_solver_eigh`) rather than damping every
direction with a Tikhonov shift.

**The step.** A step is a Runge–Kutta step: `n_stages` sample+solve evaluations,
not one gradient step.

The ansatz must be **complex-valued**: real-time evolution generates a phase, and
a real log-amplitude has no parameter that can carry it.

---

### `TDVP`

*`tachys.dynamics.tdvp`* (also exported from `tachys.dynamics`)

```python
class TDVP(*, diag_shift=0.0, mode, nbatches=1, rcond=1e-8, atol=0.0)
```

Real-time TDVP velocity. Extends `_BaseOptimizer` and is called exactly like an
optimizer — `dtheta_dt, opt_state = tdvp(E_L, opt_state, state, wf)` — but what
it returns is the physical time derivative $d\theta/dt$, not a descent direction.
Advance with $\theta + \Delta t\,\dot\theta$ (which the integrators do), never
with `apply_gradients`, whose `p - eta * g` convention would reverse the
direction of time.

| Field | Type | Description |
|-------|------|-------------|
| `diag_shift` | `float` | Tikhonov shift applied to the **kept** eigenvalues, `1 / (lambda + diag_shift)`. Default `0.0`: with the spectral truncation the solve is already well posed, and a shift biases the directions that survive. |
| `mode` | `str` | Must be `"complex"`; `"real"` raises. Static field. |
| `nbatches` | `int` | NTK sub-batching, as for the SR-family optimizers. Static field. |
| `rcond` | `float` | Relative eigenvalue cutoff — eigenvalues at or below `rcond * lambda_max` are discarded, capping the condition number of the retained subspace at `1 / rcond`. The single most important knob of a t-VMC run. |
| `atol` | `float` | Absolute floor on that cutoff. Default `0.0` (purely relative). |

`init(params)` returns `TDVPState()` (stateless; kept so the driver mirrors
`train`'s return tuple and round-trips through `tachys.checkpoint`).

---

### `evolve`

*`tachys.dynamics.real_time_evolution`* (also exported from `tachys.dynamics`)

```python
evolve(key, H, state, wf, tdvp, action, N_steps, dt, N_mc,
       integrator="rk4", t0=0.0, wandb_run=None, log_callback_fn=None,
       nsweeps=1, opt_state=None, start_step=0, tdvp_error_every=0,
       tdvp_error_rule="rect")
```

Run the t-VMC real-time evolution loop — the real-time counterpart of
`ground_state_training.train`, with the same live diagnostics table and the same
wandb / checkpoint / callback discipline. At every step the integrator performs
`n_stages` evaluations of the TDVP right-hand side (sample, local energies of
`H(t_stage)`, TDVP solve) and combines them into the parameter increment.

| Parameter | Type | Description |
|-----------|------|-------------|
| `key` | `jax.random.key` | PRNG key. |
| `H` | `_Operator` or `callable(t) -> _Operator` | Hamiltonian. For the time-dependent form only the numerical **values** of the operators' `coupling` may vary with `t`; the pytree structure, static fields and leaf shapes/dtypes must not, or every step would recompile. Checked once, up front, with an explicit error. |
| `state` | `State` | Initial Monte Carlo configuration batch. `state.lattice.Ns` sets the per-site normalizations. |
| `wf` | `WaveFunction` | Must be complex-valued. Typically the output of a ground-state `train` run. |
| `tdvp` | `TDVP` | Called as `tdvp(E_L, opt_state, state, wf)`. Passing an `SR`/`SPRING`/`MARCH` optimizer raises — they solve the imaginary-time equation. |
| `action` | `_BaseAction` | MCMC move proposal used for sampling. |
| `N_steps` | `int` | Number of time steps taken by *this* call. |
| `dt` | `float` | Time step. Cost per step is `n_stages` sample+solve evaluations. |
| `N_mc` | `int` | Number of Markov chains. |
| `integrator` | `str` or `ExplicitRK` | `"rk4"` (default), `"heun"`, or an `ExplicitRK` instance. |
| `t0` | `float` | Physical time at `start_step`. Default `0.0`. |
| `wandb_run` | optional wandb run | Logs energy, variance, acceptance, TDVP-error metrics and callback metrics every step, and checkpoints exactly as `train` does. Expected non-`None` only on `MASTER`; every rank still participates in the collective checkpoint calls. |
| `log_callback_fn` | optional `callable(state, wf, step) -> dict \| None`, or list thereof | `train`'s protocol, called once per step with the wavefunction at the **start** of the step (time `t0 + (step - start_step) * dt`) and the stage-1 batch sampled from it — the pair the reported energy is measured on, so observables computed from it cost no extra sampling. Non-`None` results are merged into the wandb log. Called on *every* rank. |
| `nsweeps` | `int` | MC sweeps per stage, passed to `sample`. Default `1`. The main lever against the warm-start lag bias, which shows up as a slow energy drift. |
| `opt_state` | optional | Pre-initialized `TDVPState` (matters only for checkpoint symmetry with `train`). |
| `start_step` | `int` | Absolute step number to resume at; offsets the printed step column, the wandb log step and the checkpoint numbering. Combine with `t0` to resume the physical time. |
| `tdvp_error_every` | `int` | If `> 0`, measure the TDVP error every this many steps with `TDVPError` and show the accumulated `R²` in the live table. `0` (default) disables it. |
| `tdvp_error_rule` | `str` | `"rect"` (default) or `"trapezoid"` — how a measurement taken every `n` steps is extended over the steps between measurements. |

**Returns** `(key, state, wf, opt_state, history)`. `history` is a
`dict[str, list]` with keys `"t"`, `"energy"` (per site), `"energy_real"`,
`"variance_per_site"`, `"acceptance"`, and — when `tdvp_error_every` is set —
`"R2"`, `"tdvp_rate"` and `"tdvp_error"` (the `TDVPError` accumulator's
per-measurement history).

---

### Integrators

*`tachys.dynamics.integrators`* (also exported from `tachys.dynamics`)

```python
class ExplicitRK(name, c, A, b, order)
Heun()      # explicit trapezoidal, order 2, 2 stages
RK4()       # classical Runge-Kutta, order 4, 4 stages
get_integrator(integrator)   # "heun" / "rk4" / an ExplicitRK instance
```

An explicit Runge–Kutta scheme defined by its Butcher tableau (`c` stage times,
`A` strictly lower-triangular coefficient rows, `b` quadrature weights). The
tableau is validated on construction: shape, row-sum condition and `sum(b) == 1`.

`step(rhs, key, t, wf, state, dt)` advances one step and returns
`(key, wf, state, ks, auxes)`, where `rhs` is
`(key, t, wf, state) -> (key, state, thetadot, aux)` and `ks`/`auxes` are the
per-stage velocity and diagnostics lists.

Two properties of the t-VMC right-hand side shape the design:

- **The chain is warm-started across stages**, never reset. Successive stage
  densities differ by `O(dt)`, so the incoming configurations are already
  `O(dt)` from equilibrium, whereas re-thermalizing at every stage would cost
  10–100× more for a *larger* bias. What remains is a lag bias of order
  `exp(-nsweeps/tau_int)`, which shows up as a slow energy drift; the cure is
  more `nsweeps`, never a chain reset.
- **Every stage draws fresh samples.** Reusing one batch for all stages of a step
  makes each `k_i` wrong by `O(dt)` — stage `i` would estimate the metric and
  force under `|psi_theta_n|²` instead of `|psi_theta_i|²` — reducing both Heun
  and RK4 to *first*-order global accuracy.

---

### `tdvp_error_rate`

*`tachys.dynamics.error`* (also exported from `tachys.dynamics`)

```python
tdvp_error_rate(wf, state, E_L, dtheta_dt, mode="complex")
```

The per-step TDVP residual rate `δs²/δt²` and its decomposition, from a single
JVP. `wf` must hold the parameters `E_L` was measured at — i.e. **before** the
integrator step; taking the JVP at the advanced parameters would put an `O(dt)`
inconsistency straight into the small residual being measured.

Writing `t_i = sum_k ΔO_ik θ̇_k` (one JVP of the ansatz with tangent `θ̇`) and
`ΔE_Li = E_Li - ⟨E_L⟩`, the three terms of

$$
\frac{\delta s^2}{\delta t^2} = \mathrm{Var}(H) + \dot\theta^T S \dot\theta - 2\,\mathrm{Im}(F)^T\dot\theta
$$

are `mean|ΔE_L|²`, `mean|t|²` and `2 Im mean[conj(t) ΔE_L]` — no `P×P` matrix
`S`, no `P`-dimensional `F`. And because
`|t + iΔE|² = |t|² + |ΔE|² - 2 Im[conj(t) ΔE]` identically, the whole rate
collapses to `mean|t + 1j ΔE|²`, which is what is evaluated: manifestly
non-negative for any `θ̇` at any sample size, and free of the catastrophic
cancellation of a difference of three `O(Var(H))` numbers.

**Returns** a `TDVPErrorEstimate` with fields `rate`, `var_H`, `quad`, `force`,
`ratio` (`force / (2 quad)`, exactly 1 when `θ̇` solves the TDVP equation — a
direct check on the velocity's normalization and sign) and `decomposed`
(`var_H + quad - force`, algebraically identical to `rate`; their difference is a
free cancellation/consistency check).

---

### `TDVPError`

*`tachys.dynamics.error`* (also exported from `tachys.dynamics`)

```python
class TDVPError(rule="rect", prefix="tdvp")
```

Accumulator for the integrated TDVP error

$$
\mathcal{R}^2(t) = \frac{1}{\sqrt{N}}\int_0^t \sqrt{\delta s^2}, \qquad
\delta s^2 = \delta t^2\left[\mathrm{Var}(\hat H) + \dot\theta^T S \dot\theta - 2\,\mathrm{Im}(F)^T\dot\theta\right]
$$

with `N = state.Ns`. Since `δs²` already carries `δt²`,
`sqrt(δs²) = dt * sqrt(rate)` and summing over steps *is* the Riemann sum of the
integral; measuring every `n` steps is the rectangle rule of width `n*dt`
(`rule="trapezoid"` averages consecutive measurements over the same interval
instead — the same cost and strictly more accurate, but not what the definition
says). Intervals are keyed on elapsed time, so a changed stride, a skipped
measurement or a short final block are all handled.

`evolve(..., tdvp_error_every=10)` builds one and feeds it a
`tdvp_error_rate` measurement every 10 steps (shown as the `R²` column of the
live table and as `history["R2"]`). The measurement uses the **first stage** of
the step — the velocity `k₁`, the batch and the local energies all at
`(t_n, theta_n)`.

| Attribute / method | Description |
|--------------------|-------------|
| `R2` | The accumulated error at the last measured step. |
| `history` | `dict[str, list]` — per-measurement `step`, `t`, `rate`, `R2`, `var_H`, `quad`, `force`, `ratio`. |
| `reset()` | Clear the accumulator and history (call before reusing the object for a second run). |
| `accumulate(step, t, Ns, est)` | Fold one `TDVPErrorEstimate` in and return the metrics to log; this is what `evolve` calls, and it works the same outside it. |

Interpretation: `δs²` is the squared Fubini–Study distance between
`exp(-iH δt)|psi(theta)>` and `|psi(theta + δt θ̇)>` to `O(δt²)` — the per-step
infidelity — so `R² √N` is the accumulated Fubini–Study angle, which upper-bounds
the angle between the exact and the variational state at time `t`. The `1/√N`
makes it intensive, since `Var(H) ~ N` for a local Hamiltonian.

Two caveats worth stating plainly. `δs²` is the residual of the *linearized*
evolution: it measures how much of `-i(H - ⟨H⟩)|psi>` lies outside the tangent
space, plus Monte Carlo and regularization error — it says nothing about the
integrator's time-discretization error, so switching Heun → RK4 will not reduce
it. And `R²` is a sum of non-negative increments, hence monotone: at long times
it is an upper bound that can be loose, so read the instantaneous `rate` (or
`rate / var_H`, the fraction of the evolution direction the manifold fails to
capture) to judge whether the state is drifting *now*.

---

## Checkpointing

### `resolve_checkpoint_settings`

*`tachys.checkpoint`*

```python
resolve_checkpoint_settings(wandb_run, N_steps, rank, MASTER)
```

Compute `(directory, save_interval_steps, max_to_keep)` on rank `MASTER` from a
wandb run's config, and broadcast the result to all ranks over multi-host
collectives. Must be called collectively by every process, including ranks
where `wandb_run` is `None`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `wandb_run` | optional wandb run | Source of `checkpoint_every` / `checkpoint_keep` config keys. Only meaningful on the source rank. |
| `N_steps` | `int` | Fallback save interval (`checkpoint_every`) when unset in `wandb_run.config`. |
| `rank` | `int` | Current process's rank (see `tachys.parallel.rank`). |
| `MASTER` | `int` | Rank designated as the config source (see `tachys.parallel.MASTER`). |

**Returns** `(dir, every, keep)`, or `(None, None, None)` if no rank has an
active `wandb_run`.

---

### `build_checkpoint_manager`

*`tachys.checkpoint`*

```python
build_checkpoint_manager(directory, save_interval_steps, max_to_keep)
```

Construct an `orbax.checkpoint.CheckpointManager` configured with a
`FixedIntervalPolicy`, so that (unlike orbax's default) it does *not* force a
checkpoint on the very first call regardless of `save_interval_steps`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `directory` | `str` | Checkpoint root directory. |
| `save_interval_steps` | `int` | Save every this many steps. |
| `max_to_keep` | `int` | Number of most-recent checkpoints to retain. |

**Returns** `orbax.checkpoint.CheckpointManager`.

---

### `get_last_step`

*`tachys.checkpoint`*

```python
get_last_step(checkpoint_dir)
```

Look up the most recent completed step number in a checkpoint directory.

| Parameter | Type | Description |
|-----------|------|-------------|
| `checkpoint_dir` | `str` | Checkpoint root directory. |

**Returns** `int | None` — the latest step, or `None` if no checkpoint exists.

---

### `save_training_checkpoint`

*`tachys.checkpoint`*

```python
save_training_checkpoint(manager, step, key, state, params, opt_state, force=False)
```

Checkpoint `params`, `key`, and the mutable parts of `state`/`opt_state` as one
atomic composite orbax checkpoint. The whole `state` pytree is saved (not just
its physical array), since foundation-model states carry extra data fields
(`system_couplings`, `system_ids`) that must round-trip too. `key` is saved as
its raw bit representation (`jax.random.key_data`), unsharded onto the global
mesh first. Must be called collectively by every process (no rank guard) so
orbax can write each host's own shards of sharded arrays.

| Parameter | Type | Description |
|-----------|------|-------------|
| `manager` | `orbax.checkpoint.CheckpointManager` | Manager returned by `build_checkpoint_manager`. |
| `step` | `int` | Step number to checkpoint under. |
| `key` | `jax.random.key` | Current PRNG key. |
| `state` | `State` | Current Monte Carlo configuration batch. |
| `params` | pytree | Wavefunction parameters (`wf.params`). |
| `opt_state` | pytree | Optimizer state. |
| `force` | `bool` | Force a save even outside the manager's save interval (e.g. on the final training step). Default `False`. |

**Returns** whatever `manager.save(...)` returns (orbax's save future / bool).

---

### `load_checkpoint`

*`tachys.checkpoint`*

```python
load_checkpoint(checkpoint_dir, state_template, opt_state_template, params_template=None, step=None)
```

Restore `params`, `opt_state`, `state` and `key` from a checkpoint directory
written by `save_training_checkpoint`. `state_template` and
`opt_state_template` supply the structural pieces that aren't serialized
(static fields like `state.lattice`, and the `opt_state` `NamedTuple` type);
every data field of the restored objects is overwritten with the checkpointed
values. Restoring is portable across device topologies: everything is placed
onto the *current* global `mesh` (from `tachys.parallel`) rather than the
sharding recorded at save time. `params`/`opt_state`/`key` are restored fully
replicated; `state` is restored partitioned along the mesh's `'i'` axis.

| Parameter | Type | Description |
|-----------|------|-------------|
| `checkpoint_dir` | `str` | Checkpoint root directory. |
| `state_template` | `State` | Structural template for restoring `state` (supplies static fields). |
| `opt_state_template` | pytree | Structural template for restoring `opt_state` (e.g. from `optimizer.init(params)`). |
| `params_template` | optional pytree | Structural template for `params`. Omit to recover params as a plain dict inferred from checkpoint metadata (fine when `wf.params` is already a plain dict), at the cost of a suppressed sharding-fallback warning. |
| `step` | optional `int` | Step to restore. Defaults to the manager's latest step. |

**Returns** `(params, opt_state, state, key)`.

---

## Parallelism

*`tachys.parallel`*

Module-level constants describing the current JAX device topology, computed
once at import time and used throughout `tachys` for `shard_map`-based
multi-device/multi-host parallelism.

| Name | Type | Description |
|------|------|-------------|
| `mesh` | `jax.sharding.Mesh` | Device mesh over all `jax.devices()`, with a single named axis `'i'`. Used as the sharding mesh for every `shard_map`/`NamedSharding` call in `tachys` (Monte Carlo sampling, optimizers, checkpointing). |
| `n_devices` | `int` | Total number of devices, `len(jax.devices())`. |
| `rank` | `int` | Current process's index, `jax.process_index()`. `0` on a single-process run. |
| `MASTER` | `int` | Rank designated to own single-writer responsibilities (logging, wandb, checkpoint config). Always `0`. |

---

### `all_unshard`

```python
all_unshard(pytree)
```

Force every leaf of `pytree` onto a fully replicated sharding (`P()` over
`mesh`) — every device holds a full copy. Used to promote host-local arrays
(e.g. a PRNG key produced by plain `jax.random.split`) to a proper multi-host
global array before checkpointing.

| Parameter | Type | Description |
|-----------|------|-------------|
| `pytree` | pytree of `jax.Array` | Arrays to replicate. |

**Returns** the same pytree with every leaf's sharding constrained to `P()`.

---

### `promote_to_pytree`

```python
promote_to_pytree(f)
```

Decorator that lifts a function operating on a single array to one that
`jax.tree.map`s it over an arbitrary pytree. Used to define `hard_shard`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `f` | `callable(array) -> array` | Function to lift. |

**Returns** `callable(pytree) -> pytree`.

---

### `hard_shard`

```python
hard_shard(array)
```

Slice the leading axis of every leaf of a pytree into `n_devices` equal
contiguous chunks and keep only the chunk belonging to the current `rank`
— an explicit (non-`jax.jit`) host-side partition, distinct from
`shard_map`'s device-level sharding. Requires the leading axis length to be
divisible by `n_devices`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `array` | pytree of `jax.Array` | Data to partition; leading axis is split. |

**Returns** the same pytree, restricted to this rank's local chunk.

---

## Optimizers

### `_BaseOptimizer`

*`tachys.optimizer.optimizers`*

```python
class _BaseOptimizer(diag_shift, mode, nbatches=1)
```

Abstract base for all natural-gradient (SR-family) optimizers. Extends
`flax.struct.PyTreeNode`. Subclass it and implement `init(params)` and
`update(E_L, opt_state, state, wf, weights=None)`. Handles the `shard_map`
dispatch (`__call__`) so subclasses only need to implement per-shard logic.

| Field | Type | Description |
|-------|------|-------------|
| `diag_shift` | `float` | Diagonal (Tikhonov) regularization added to the NTK before solving. |
| `mode` | `str` | `"complex"`: the real and imaginary parts of `log ψ` enter the kernel as separate rows, so a parameter-dependent phase is optimized too. `"real"`: only `Re log ψ = log|ψ|` enters, which is exact when the phase does not depend on the parameters. The parameters themselves must be real in both modes (complex-dtype parameters give a zero update). Static (non-pytree) field. |
| `nbatches` | `int` | Number of sub-batches the NTK assembly splits the Monte Carlo batch into (trades memory for extra compute). Static field, default `1`. |

| Member | Type | Description |
|--------|------|-------------|
| `init(params)` | `pytree → OptimizerState` | Build the initial optimizer state for a given parameter pytree. |
| `update(E_L, opt_state, state, wf, weights=None)` | `→ (updates, new_opt_state)` | Per-shard update rule. Override in subclasses. |
| `__call__(E_L, opt_state, state, wf, weights=None)` | `→ (updates, new_opt_state)` | JIT-compiled, `shard_map`-wrapped entry point; dispatches to `update`. |

**Returns** (of `__call__`) `(updates, new_opt_state)`, where `updates` is a
pytree matching `wf.params`, meant to be passed to `wf.apply_gradients`.

---

### `SR`

*`tachys.optimizer.optimizers`* (also exported from `tachys.optimizer`)

```python
class SR(diag_shift, mode, nbatches=1)
```

Stochastic Reconfiguration: the natural-gradient update obtained from the
neural tangent kernel (NTK) $S$ of the wavefunction and the energy force
vector $\boldsymbol\varepsilon$,

$$
(S + \lambda I)\,\delta\theta = \boldsymbol\varepsilon, \qquad
\varepsilon_i = \frac{2\,(E_{L,i} - \bar E_L)^{*}}{\sqrt{N_{mc}}}
$$

solved via a Cholesky decomposition (`tachys.optimizer._kernels.linear_solver_cholesky`),
then mapped back to parameter space with a VJP through `wf.apply_fn`.

`init(params)` returns `SRState()` (stateless). Calling the instance computes
one SR update from a batch of local energies.

---

### `SRState`

*`tachys.optimizer.optimizers`* (also exported from `tachys.optimizer`)

```python
class SRState()
```

Empty `NamedTuple` — `SR` carries no state between steps.

---

### `SPRING`

*`tachys.optimizer.optimizers`* (also exported from `tachys.optimizer`)

```python
class SPRING(diag_shift, mode, nbatches=1, *, mu=0.9)
```

SR with Projected Nesterov-style momentum. Folds a JVP-based momentum
correction into the force vector before solving, then adds momentum to the
resulting parameter update:

$$
\delta\theta_t = \delta\theta_t^{\mathrm{SR}} + \mu\,\delta\theta_{t-1}
$$

| Field | Type | Description |
|-------|------|-------------|
| `mu` | `float` | Momentum coefficient. Default `0.9`. |

(Inherits `diag_shift`, `mode`, `nbatches` from `_BaseOptimizer`.)

`init(params)` returns `SPRINGState(old_updates=zeros_like(params))`.

---

### `SPRINGState`

*`tachys.optimizer.optimizers`* (also exported from `tachys.optimizer`)

```python
class SPRINGState(old_updates)
```

| Field | Type | Description |
|-------|------|-------------|
| `old_updates` | pytree matching `wf.params` | Parameter update from the previous step, used for the momentum term. |

---

### `MARCH`

*`tachys.optimizer.optimizers`* (also exported from `tachys.optimizer`)

```python
class MARCH(diag_shift, mode, nbatches=1, *, mu=0.95, beta=0.995)
```

SPRING augmented with an adaptive second-moment preconditioner (analogous to
Adam's second moment): an exponential moving average $V$ of squared parameter
update differences is maintained and its bias-corrected value scales both the
NTK and the final update,

$$
V_t = \beta V_{t-1} + (1-\beta)\,\lvert \delta\theta_{t-1} - \delta\theta_{t-2} \rvert^2,
\qquad
\delta\theta_t = \frac{\delta\theta_t^{\mathrm{SR}}}{\sqrt{\hat V_t} + \epsilon} + \mu\,\delta\theta_{t-1}
$$

| Field | Type | Description |
|-------|------|-------------|
| `mu` | `float` | Momentum coefficient. Default `0.95`. |
| `beta` | `float` | Exponential-moving-average decay for the second moment `V`. Default `0.995`. |

(Inherits `diag_shift`, `mode`, `nbatches` from `_BaseOptimizer`.)

`init(params)` returns `MARCHState(old_updates=zeros_like(params), V=ones_like(params), t=0)`.

---

### `MARCHState`

*`tachys.optimizer.optimizers`* (also exported from `tachys.optimizer`)

```python
class MARCHState(old_updates, V, t)
```

| Field | Type | Description |
|-------|------|-------------|
| `old_updates` | pytree matching `wf.params` | Parameter update from the previous step. |
| `V` | pytree matching `wf.params` | Exponential moving average of squared update differences (uncorrected). |
| `t` | `int32` | Step counter, used for bias correction of `V`. |

---

### `linear_decay`

*`tachys.optimizer.lr_schedules`* (also exported from `tachys.optimizer`)

```python
linear_decay(eta0, eta_final, N_steps)
```

Linear learning-rate decay from `eta0` to `eta_final` over `N_steps`, as a
function of the *absolute* step — so it decays correctly across resumes when
the step passed in is offset by `start_step`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `eta0` | `float` | Initial learning rate. |
| `eta_final` | `float` | Final learning rate, reached at `step = N_steps` and held thereafter. |
| `N_steps` | `int` | Number of steps over which to decay. |

**Returns** `callable(step: int) -> float`.

---

### `shifted_cosine_decay`

*`tachys.optimizer.lr_schedules`* (also exported from `tachys.optimizer`)

```python
shifted_cosine_decay(init_value, decay_steps, min_value=None)
```

Cosine-decay learning-rate schedule (via `optax.cosine_decay_schedule`),
shifted so its floor is `min_value` instead of `0`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `init_value` | `float` | Learning rate at step `0`. |
| `decay_steps` | `int` | Number of steps over which the cosine decay runs. |
| `min_value` | optional `float` | Floor value the schedule decays to. Defaults to `init_value / 10`. |

**Returns** `callable(step: int) -> float`.

---

### Low-level SR kernels

*`tachys.optimizer._kernels`*

Building blocks used internally by `SR`/`SPRING`/`MARCH`, exposed for advanced
use (e.g. implementing a custom SR-family optimizer, or unit-testing the NTK
pipeline directly).

#### `linear_solver_cholesky`

```python
linear_solver_cholesky(ntk, eps, diag_shift, mode="complex")
```

Solve the regularized SR linear system $(S + \lambda I)\,\delta\theta = \varepsilon$
via Cholesky decomposition. In `mode="complex"`, the complex linear system is
solved as an equivalent real `2M × 2M` block system (real/imaginary parts) using
two nested Cholesky solves (Schur complement). Degrades to a no-op (returns
`0`) for any row where the solve produces non-finite values, guarding against
`jnp.linalg.cholesky`'s under-`jit` behavior of silently filling non-PD rows
with NaN instead of raising.

| Parameter | Type | Description |
|-----------|------|-------------|
| `ntk` | `jax.Array` | `mode="real"`: shape `(..., M, M)`, real SPD. `mode="complex"`: shape `(..., M, M, 2, 2)` block matrix. |
| `eps` | `jax.Array` | Force vector, shape `(..., M)`; complex in `mode="complex"`. |
| `diag_shift` | `float` | Regularization added to the diagonal. |
| `mode` | `str` | `"real"` or `"complex"`. Default `"complex"`. |

**Returns** `(..., M)` real, or `(..., 2*M)` real `[u, v]` (with solution
`x = u + iv`) in `mode="complex"`.

---

#### `linear_solver_eigh`

```python
linear_solver_eigh(ntk, eps, diag_shift, mode="complex", rcond=1e-8, atol=0.0)
```

Spectrally-truncated (pseudo-inverse) solver for the same system — a drop-in
replacement for `linear_solver_cholesky` with the same arguments and the same
return layout. Diagonalizes the kernel and inverts it only on the eigenvectors
whose eigenvalue clears `cutoff = max(rcond * lambda_max, atol)`, projecting the
rest away. Used by `tachys.dynamics.TDVP`: in real-time evolution the kernel is
genuinely rank deficient (centering alone puts exact zero modes in the spectrum,
and `2M > n_params` makes it singular by construction), and Tikhonov damping
distorts the well-resolved directions instead of removing the unresolved ones.

| Parameter | Type | Description |
|-----------|------|-------------|
| `ntk` | `jax.Array` | `mode="real"`: `(..., M, M)`. `mode="complex"`: `(..., M, M, 2, 2)`. |
| `eps` | `jax.Array` | Force vector, shape `(..., M)`; complex in `mode="complex"`. |
| `diag_shift` | `float` | Tikhonov shift applied to the **kept** eigenvalues, `1 / (lambda + diag_shift)`. Because `diag_shift * I` is isotropic it commutes with the eigendecomposition, so the matrix is never modified. |
| `mode` | `str` | `"real"` or `"complex"`. Default `"complex"`. |
| `rcond` | `float` | Relative eigenvalue cutoff. Relative rather than absolute because the kernel's scale varies by orders of magnitude with ansatz, system size and step, whereas `1 / rcond` is exactly the condition number the retained subspace is capped at. |
| `atol` | `float` | Absolute floor on the cutoff. Default `0.0`. |

**Returns** `(..., M)` real, or `(..., 2*M)` real `[u, v]` in `mode="complex"` —
matching `linear_solver_cholesky`.

The keep mask reads the **raw** spectrum, before `diag_shift` is applied, so the
two regularizers stay orthogonal: `rcond` chooses the retained subspace,
`diag_shift` softens the amplification inside it. (Masking `lambda + diag_shift`
instead would let a large enough shift silently switch the truncation off.) The
mask is `lambda > cutoff`, not `|lambda| > cutoff`: the kernel is a Gram matrix,
so a negative eigenvalue is roundoff on a signal-free direction, and inverting it
would flip the update along that direction and amplify it by `1 / |lambda|`.
Non-finite input degrades to a zero update, as in the Cholesky path.

Note that a hard truncation makes the solution discontinuous in the parameters
whenever an eigenvalue crosses the threshold — harmless for a fixed-step
integrator, but it would corrupt an embedded error estimate used for step-size
control.

---

#### `ntk_parallel_fn`

```python
ntk_parallel_fn(state, wf, nbatches, mode, V=None)
```

Assemble the full `(N_mc × N_mc)` neural tangent kernel matrix by distributing
pairwise per-batch Jacobian contractions across devices and reducing with
`psum`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `state` | `State` | Local shard of the Monte Carlo batch. |
| `wf` | `WaveFunction` | Wavefunction whose Jacobian w.r.t. parameters is contracted. |
| `nbatches` | `int` | Number of sub-batches to split the local batch into. |
| `mode` | `str` | `"real"` or `"complex"`. |
| `V` | optional pytree matching `wf.params` | MARCH's bias-corrected second-moment preconditioner. |

**Returns** the full NTK: shape `(N_mc, N_mc)` (real) or `(N_mc, N_mc, 2, 2)`
(complex).

---

#### `center_ntk`

```python
center_ntk(ntk, weights, state)
```

Subtract row, column, and global means from the NTK (the centering step
equivalent to centering the Jacobian before contraction). Uses per-system
means when `state` is a `FoundationState`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `ntk` | `jax.Array` | Fully gathered/replicated NTK, from `ntk_parallel_fn`. |
| `weights` | optional `jax.Array` | Per-sample reweighting (e.g. importance weights). Not yet supported together with `FoundationState`. |
| `state` | `State` | Determines whether per-system (`FoundationState`) or global centering is used. |

**Returns** the centered NTK, same shape as `ntk`.

---

#### `compute_ntk`

```python
compute_ntk(state, wf, mode, weights=None, V=None, nbatches=1)
```

Full NTK pipeline: `ntk_parallel_fn` → `center_ntk` → optional `sqrt(weights)`
row/column scaling.

| Parameter | Type | Description |
|-----------|------|-------------|
| `state` | `State` | Local shard of the Monte Carlo batch. |
| `wf` | `WaveFunction` | Wavefunction. |
| `mode` | `str` | `"real"` or `"complex"`. |
| `weights` | optional `jax.Array` | Per-sample reweighting. |
| `V` | optional pytree matching `wf.params` | MARCH preconditioner. |
| `nbatches` | `int` | Sub-batch count for the pairwise Jacobian assembly. Default `1`. |

**Returns** the centered (and optionally reweighted) NTK.

---

#### `center_sr_solution`

```python
center_sr_solution(sr_solution, state, mode, weights)
```

Center the linear-solve output before the final VJP step (mirrors
`center_ntk`'s centering, applied to the solution vector rather than the
kernel). Uses per-system means when `state` is a `FoundationState`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `sr_solution` | `jax.Array` | Output of `linear_solver_cholesky`. |
| `state` | `State` | Determines per-system vs. global centering. |
| `mode` | `str` | `"real"` or `"complex"`. |
| `weights` | optional `jax.Array` | Per-sample reweighting. |

**Returns** the centered solution, reshaped to `(N_mc, 2)` in `mode="complex"`
before the caller's VJP.

---

## Collectives

Sharded-mesh reduction helpers for grouping per-sample quantities by system. Meant to be called
inside a `shard_map` over mesh axis `'i'`: each shard computes a local per-group reduction, then
`jax.lax.psum` combines the shards so every shard ends up with the same, fully-reduced result —
groups whose elements are split across shards are still reduced correctly. Used by foundation-
model training to average quantities (e.g. local energies) per-system rather than over the whole
mixed batch.

### `grouped_sum`

*`tachys.lattice.foundation.collectives`*

```python
grouped_sum(x, y, K, axis=0)
```

Sums `x` into `K` groups given by `y`, reduced across the sharded mesh axis `'i'`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `jax.Array` | Local shard, any shape with `x.shape[axis] == y.shape[0]`. |
| `y` | `jax.Array` | Local shard of integer group labels in `[0, K)`. |
| `K` | `int` | Number of groups. Must be a static (non-traced) Python `int`, since it is used as `segment_sum`'s `num_segments`. |
| `axis` | `int` | Axis of `x` indexed by `y`; replaced by `K` in the output. Default `0`. |

**Returns** `jax.Array` like `x` but with size `K` along `axis`, replicated over the mesh.

---

### `grouped_mean`

*`tachys.lattice.foundation.collectives`*

```python
grouped_mean(x, y, K, axis=0, broadcast=False)
```

Averages `x` within each of `K` groups given by `y`. Same contract as `grouped_sum`, but averages
within each group instead of summing. Counts are reduced the same way (per-shard `segment_sum`
then `psum`) so groups split across shards are still averaged correctly.

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `jax.Array` | Local shard, any shape with `x.shape[axis] == y.shape[0]`. |
| `y` | `jax.Array` | Local shard of integer group labels in `[0, K)`. |
| `K` | `int` | Number of groups (static Python `int`). |
| `axis` | `int` | Axis of `x` indexed by `y`. Default `0`. |
| `broadcast` | `bool` | If `False` (default), return the reduced `(..., K, ...)` array of per-group means, replicated over the mesh. If `True`, return an array shaped like `x` instead, with each element replaced by its own group's mean — e.g. to center per-group values (such as per-system local energies) via `x - grouped_mean(x, y, K, axis, True)`. Note this changes the sharding of the result along `axis` from replicated to matching `x`'s local shard, so the `shard_map` call site's `out_specs` must be updated accordingly (e.g. `P('i')` instead of `P()`). |

**Returns** `jax.Array`. Shape `(..., K, ...)` if `broadcast=False`, or shaped like `x` if
`broadcast=True`.

---

## Utilities

*`tachys.utils`*

### `same_treedef`

```python
same_treedef(tree1, tree2)
```

Check whether two pytrees have identical structure (types, nesting, and static
fields). Compares `repr(jax.tree.structure(...))` rather than using
`PyTreeDef.__eq__` directly, since recent JAX/Flax versions changed `__eq__` to
ignore the registered node type (e.g. `Splus == Sminus` would compare equal
under `__eq__`). Useful when writing a custom `_BaseAction` to verify a
proposed new state has the same structure as the original.

| Parameter | Type | Description |
|-----------|------|-------------|
| `tree1`, `tree2` | pytree | Trees to compare. |

**Returns** `bool`.

---

### `same_treedef_and_avals`

```python
same_treedef_and_avals(tree1, tree2)
```

Like `same_treedef`, but additionally requires every leaf to have matching
shape and dtype.

| Parameter | Type | Description |
|-----------|------|-------------|
| `tree1`, `tree2` | pytree | Trees to compare. |

**Returns** `bool`.

---

### `as_column`

```python
as_column(x)
```

Reshape a 1-D array to a column vector `(N, 1)`; leaves arrays of other ranks
unchanged.

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | array-like | Input array. |

**Returns** `jnp.ndarray`.

---

## Exact diagonalization

### `build_sparse_hamiltonian`

*`tachys.lattice.exact_diag`*

```python
build_sparse_hamiltonian(state_full_hilbert, H, pack)
```

Assemble the sparse matrix of `H` in the basis enumerated by
`state_full_hilbert`. Split out of `exact_diag` so the matrix itself is
reachable — needed by anything that wants more than the extremal eigenpairs,
e.g. exact real-time propagation `expm(-1j * H * t) @ psi` for validating
`tachys.dynamics`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `state_full_hilbert` | `State` | Batch containing every basis state in the Hilbert space. |
| `H` | `callable` | Hamiltonian operator. Must return `DiagOffdiagResult`. |
| `pack` | `callable` | Maps a state batch to a 1-D integer index array. Must be injective. |

**Returns** `(mat, sorted_active)` — a `scipy.sparse.csr_array` of shape
`(n_active_states, n_active_states)`, and the sorted `pack` indices in the order
the matrix rows/columns use, so
`np.searchsorted(sorted_active, pack(some_state))` maps any state back to its
matrix index.

---

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
</content>
