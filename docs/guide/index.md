# User Guide

The {doc}`../quickstart` builds a VMC script from five components. Each page of
this guide covers one of them: what the library provides, and how to build your
own.

| Component | Page | Main objects |
|---|---|---|
| Lattice | {doc}`lattices` | `square`, `Lattice.create`, `lattice.bonds` |
| Hamiltonian | {doc}`hamiltonians` | `Sz`, `Splus`, `Cup_dag`, `heisenberg_hamiltonian` |
| Initial configurations | {doc}`configurations` | `SpinState`, `FermionState`, `init_config_fixed_magn` |
| Wavefunction | {doc}`wavefunctions` | `WaveFunction`, `SpinViT`, `FermionicTransformer` |
| Monte Carlo move | {doc}`sampling` | `sample`, `BondExchange`, `SpinFlip` |
| Optimization step | {doc}`optimization` | `SR`, `MARCH`, `train` |

Code on these pages continues from the quickstart scripts: names such as
`lattice`, `key`, `N_mc`, `state` and `wf` refer to the objects defined there.

```{toctree}
:hidden:

lattices
hamiltonians
configurations
wavefunctions
sampling
optimization
```
