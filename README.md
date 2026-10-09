<p align="center">
  <img src="https://raw.githubusercontent.com/tachys-dev/tachys/main/docs/_static/logo.svg" width="285" height="48" alt="tachys logo">
</p>

# Variational Monte Carlo for quantum many-body systems

[![Tests](https://github.com/tachys-dev/tachys/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/tachys-dev/tachys/actions/workflows/tests.yml?query=branch%3Amain)
[![PyPI version](https://img.shields.io/pypi/v/tachys)](https://pypi.org/project/tachys/)

**tachys** is a variational Monte Carlo library developed in JAX with one
central aim: bringing neural-network quantum states (NQS) to realistic quantum
materials. To reach the system sizes that real materials demand, the code is
optimized to scale across large GPU clusters and has been benchmarked on
systems with up to 1,000 electrons, a scale already meaningful for studying
unconventional superconductors. With the same focus on realistic materials,
tachys can treat itinerant fermions and localized spins within a single
Hamiltonian, as required by Kondo lattice models of heavy-fermion compounds.
It also supports Foundation NQS, in which one network is trained across an
entire family of Hamiltonians, allowing the parameter space relevant to a given
material to be explored in a single optimization.

**Documentation:** https://tachys.org/quickstart.html

## Installation

**CPU:**
```bash
pip install tachys
```

**CUDA (GPU):**
```bash
pip install "tachys[cuda13]"    # CUDA 13
pip install "tachys[cuda12]"    # CUDA 12
```

These install JAX together with the CUDA 13 or CUDA 12 libraries, so the
machine needs only the NVIDIA driver. Choose the version that the driver
supports: the header of `nvidia-smi` shows the highest one.

### Install from source

```bash
git clone https://github.com/tachys-dev/tachys.git
cd tachys
pip install .
```

### Development install

To install in editable mode (changes to the source are reflected immediately without reinstalling):

```bash
pip install -e ".[test]"            # CPU
pip install -e ".[cuda13,test]"     # CUDA 13
pip install -e ".[cuda12,test]"     # CUDA 12
pytest tests/
```

## Citing tachys

If you use tachys in your research, please cite it:

```bibtex
@software{tachys,
  author = {Rende, Riccardo and Viteritti, Luciano Loris},
  title  = {tachys: Variational Monte Carlo for quantum many-body systems},
  url    = {https://github.com/tachys-dev/tachys},
  year   = {2026},
}
```

## License

Copyright 2026 The Simons Foundation, Inc.

Licensed under the [Apache License, Version 2.0](https://github.com/tachys-dev/tachys/blob/main/LICENSE).
