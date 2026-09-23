<p align="center">
  <img src="https://raw.githubusercontent.com/tachys-dev/tachys/main/docs/_static/logo.svg" width="285" height="48" alt="tachys logo">
</p>

# tachys

**tachys** is a JAX library for finding the ground states of quantum lattice
models with neural-network wavefunctions. Build a Hamiltonian by composing
spin and fermionic operators, use any Flax network as the wavefunction, and
optimize it with variational Monte Carlo — the same code runs on a laptop and
across a multi-GPU cluster. It also trains foundation models: a single network
covering a whole family of Hamiltonians, rather than one run per coupling.

## Installation

**CPU:**
```bash
pip install tachys
```

**CUDA (GPU):**
```bash
pip install "tachys[cuda]"
```

This installs JAX with CUDA 12 support via the `cuda` optional dependency.

### Install from source

```bash
git clone https://github.com/tachys-dev/tachys.git
cd tachys
pip install .
```

### Development install

To install in editable mode (changes to the source are reflected immediately without reinstalling):

```bash
pip install -e ".[test]"          # CPU
pip install -e ".[cuda,test]"     # CUDA
pytest tests/
```

## Developers

- Riccardo Rende
- Luciano Loris Viteritti
