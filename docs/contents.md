---
html_theme.sidebar_secondary.remove: true
---

# Documentation

```{admonition} Homepage
:class: seealso
Looking for the project overview? Head back to the <a href="index.html">Tachys homepage</a>.
```

**Tachys** is a JAX library for quantum many-body physics on lattices.
It treats operators as first-class callables — compose them algebraically,
apply them to batched states, and run exact diagonalization on any Hamiltonian
you can write.

## Installation

```bash
# CPU-only JAX
pip install jax

# GPU (CUDA 12)
pip install "jax[cuda12]"

# Tachys
pip install tachys
```

```{toctree}
:maxdepth: 1
:caption: Getting Started

quickstart
```

```{toctree}
:maxdepth: 1
:caption: Documentation

concepts
foundation_models
api
```
</content>
</invoke>
