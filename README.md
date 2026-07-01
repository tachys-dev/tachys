<p align="center">
  <img src="docs/_static/logo.svg" width="285" height="48" alt="tachys logo">
</p>

# tachys

TO CHECK:

 - NTK
 - BondExchange

## Installation

This is a **private repository**, so `pip` needs to authenticate to GitHub before it can fetch the code. The two supported ways to do that are SSH (recommended if you already have an SSH key registered with GitHub) or an HTTPS personal access token.

### Install directly with pip (no local clone)

Since the repo is private, plain `pip install git+https://...` will fail with an authentication error unless you provide credentials. Use one of the following instead.

**Via SSH** (requires your SSH key to be added to your GitHub account and access to the repo):

```bash
pip install "git+ssh://git@github.com/riccardo-rende/tachys.git"
```

**Via HTTPS with a personal access token** (create one under GitHub Settings → Developer settings → Personal access tokens, with at least `repo` scope):

```bash
pip install "git+https://<YOUR_TOKEN>@github.com/riccardo-rende/tachys.git"
```

To install with the CUDA extra, append `#egg=tachys[cuda]` to either URL, e.g.:

```bash
pip install "git+ssh://git@github.com/riccardo-rende/tachys.git#egg=tachys[cuda]"
```

### Clone and install

Clone the repository and install with `pip`:

```bash
git clone git@github.com:riccardo-rende/tachys.git
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
- [einops](https://github.com/arogozhnikov/einops)
