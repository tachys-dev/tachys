---
html_theme.sidebar_secondary.remove: true
---

# Documentation

```{admonition} Homepage
:class: seealso
Looking for the project overview? Head back to the <a href="index.html">tachys homepage</a>.
```

```{admonition} Benchmarks
:class: tip
Reference variational energies for frustrated lattice models — pick a model, a
lattice geometry and a size on the <a href="benchmarks.html">benchmarks page</a>.
```

**tachys** is a JAX library for finding the ground states of quantum lattice
models with neural-network wavefunctions. Build a Hamiltonian by composing
spin and fermionic operators, use any Flax network as the wavefunction, and
optimize it with variational Monte Carlo — the same code runs on a laptop and
across a multi-GPU cluster. It also trains foundation models: a single network
covering a whole family of Hamiltonians, rather than one run per coupling.

## Installation

```bash
pip install tachys            # CPU
pip install "tachys[cuda]"    # GPU, with JAX for CUDA 12
```

```{toctree}
:maxdepth: 1
:caption: Getting Started

quickstart
concepts
```

```{toctree}
:maxdepth: 1
:caption: Documentation

guide/index
ground_state_training
foundation_models
parallel
api
```

```{toctree}
:maxdepth: 1
:caption: Additional Resources

resources/vit_wavefunction
resources/sbp_wavefunction
resources/pfaffian_wavefunction
```
