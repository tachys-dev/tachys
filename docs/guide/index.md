# User Guide

The {doc}`../quickstart` builds a VMC script from five components. Each page of
this guide covers one of them: what the library provides, and how to build your
own.

| Component | Page | Main objects |
|---|---|---|
| Lattice | {doc}`lattice` | `square`, `Lattice.create`, `lattice.bonds` |
| Hamiltonian | {doc}`hamiltonian` | `Sz`, `Splus`, `Cup_dag`, `heisenberg_hamiltonian` |
| Initial configuration | {doc}`configuration` | `SpinState`, `FermionState`, `init_config_fixed_magn` |
| Wavefunction | {doc}`wavefunction` | `WaveFunction`, `SpinViT`, `FermionicTransformer` |
| Monte Carlo move | {doc}`sampling` | `sample`, `BondExchange`, `SpinFlip` |
| Optimization step | {doc}`optimization` | `SR`, `MARCH`, `train` |

Each code cell on these pages can be read on its own: it defines every name it
uses, except those of the quickstart scripts, such as `lattice`, `key`, `N_mc`,
`state` and `wf`, which refer to the objects defined there.

```{toctree}
:hidden:

lattice
hamiltonian
configuration
wavefunction
sampling
optimization
```
