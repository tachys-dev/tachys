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

**tachys** is a variational Monte Carlo library developed in JAX with one
central aim: bringing neural-network quantum states (NQS) to realistic quantum
materials. To reach the system sizes that real materials demand, the code is
engineered to scale across {doc}`large GPU clusters <parallel>` and has been
benchmarked on systems with up to 1,000 electrons, a scale already meaningful
for studying unconventional superconductors. With the same focus on realistic
materials, tachys can treat itinerant fermions and localized spins within a
single Hamiltonian, as required by
{doc}`Kondo lattice models <resources/pfaffian_wavefunction>` of heavy-fermion
compounds. It also supports {doc}`Foundation NQS <foundation_models>`, in which
one network is trained across an entire family of Hamiltonians, allowing the
parameter space relevant to a given material to be explored in a single
optimization.

```{toctree}
:maxdepth: 1
:caption: Getting Started

installation
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
