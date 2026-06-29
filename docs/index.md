# Tachys

**Tachys** is a JAX library for quantum many-body physics on lattices.
It treats operators as first-class callables — compose them algebraically,
apply them to batched states, and run exact diagonalization on any Hamiltonian
you can write.

::::{grid} 3
:gutter: 2
:margin: 4 4 0 0

:::{grid-item-card} Functional operator algebra
Operators compose with `+` and `*`. Any result is itself callable — there is no
boundary between a primitive operator and a full Hamiltonian.
:::

:::{grid-item-card} JAX-native state pytrees
States extend `flax.struct.PyTreeNode`. They flow through `jit`, `vmap`, and
`grad` without any wrapping.
:::

:::{grid-item-card} One-call exact diagonalization
Generate the full Hilbert space, build the sparse matrix, and compute the lowest
eigenvalues — all in a single function call.
:::

::::

## Installation

```bash
# CPU-only JAX
pip install jax

# GPU (CUDA 12)
pip install "jax[cuda12]"

# Tachys
pip install tachys
```

## Getting started

```{toctree}
:maxdepth: 1
:caption: Getting Started

quickstart
```

```{toctree}
:maxdepth: 1
:caption: Documentation

concepts
api
examples
```
