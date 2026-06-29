# tachys

## Installation

Clone the repository and install with `pip`:

```bash
git clone <repo-url>
cd tachys
```

**CPU:**
```bash
pip install .
```

**CUDA (GPU):**
```bash
pip install ".[cuda]"
```

This installs JAX with CUDA 12 support via the `cuda` optional dependency.

### Development install

To install in editable mode (changes to the source are reflected immediately without reinstalling):

```bash
pip install -e .          # CPU
pip install -e ".[cuda]"  # CUDA
```

## Requirements

- Python >= 3.9
- [JAX](https://github.com/google/jax) 0.7.2
- [Flax](https://github.com/google/flax) 0.12.0
- NumPy
- SciPy
