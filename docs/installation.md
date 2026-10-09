# Installation

tachys needs Python 3.12 or later. pip installs it together with JAX for CPUs,
Flax and the other dependencies, at the versions that tachys is tested with:

```bash
pip install tachys
```

## GPU

On Linux machines with NVIDIA GPUs, the extras `cuda13` and `cuda12` install
JAX with the CUDA 13 or CUDA 12 libraries:

```bash
pip install "tachys[cuda13]"    # CUDA 13
pip install "tachys[cuda12]"    # CUDA 12
```

The CUDA libraries come with the package, so the machine needs only the NVIDIA
driver. Choose the CUDA version that the driver supports: the header of
`nvidia-smi` shows the highest one.

## Checking the installation

Importing tachys prints a banner with the versions of tachys, JAX and Flax,
and with the devices that JAX runs on:

```bash
python -c "import tachys"
```

On a GPU, the devices are listed as

```text
1 × GPU  (NVIDIA H100)
1 node  ·  1 GPU/node
```

`CPU` in place of `GPU` means that JAX did not find the GPU: check that tachys
was installed with a CUDA extra, for a CUDA version that the driver supports.
tachys runs one process per GPU, and a process that sees several GPUs cannot
import it ({doc}`parallel`), so on a machine with several GPUs the check makes
one of them visible:

```bash
CUDA_VISIBLE_DEVICES=0 python -c "import tachys"
```

## Other extras

The extra `wandb` installs the packages used by {doc}`ground_state_training`.
Extras combine:

```bash
pip install "tachys[cuda13,wandb]"
```

## From source

An editable install from the repository takes changes to the source into
account without reinstalling; the extra `test` adds pytest, for the tests:

```bash
git clone https://github.com/tachys-dev/tachys.git
cd tachys
pip install -e ".[test]"            # ".[cuda13,test]" on GPUs
pytest tests/
```
