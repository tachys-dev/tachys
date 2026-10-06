# Parallel Jobs

The same tachys script runs on one GPU or on many. The $N_{mc}$ Markov chains
are divided evenly among the GPUs: each GPU advances its own chains and
evaluates their local energies and log-derivatives, while the energy and the
parameter update are computed from the samples of all GPUs, as on a single
device. This page shows how to launch such runs with Slurm, the job scheduler
of many computing clusters, on one node (one computer of the cluster) or on
several.

## Processes and GPUs

A parallel run consists of one process per GPU, all executing the same script.
On a cluster, `srun`, the launcher of Slurm, starts them; no MPI library is
needed. When the script imports tachys, the processes connect to each other and
each takes one GPU of its node; `jax.devices()` then lists the GPUs of all the
processes, and the chains are divided among them. Three requirements follow:

- **One process per GPU.** Each node runs as many processes as it has GPUs;
  importing tachys in a process that sees several GPUs raises an error.
- **`N_mc` is a multiple of the number of GPUs**, `n_devices` in
  `tachys.parallel`, so that every GPU holds the same number of chains.
- **tachys is imported before any JAX array is created**, since JAX can connect
  the processes only before its first computation.

Only the master process, for which `rank == MASTER`, prints, so the output is
not repeated by every process. Operations to be performed once, such as
writing a file, must likewise be restricted to it; `train`, `sample` and the
other functions of tachys, instead, must be called by every process. With the objects of the {doc}`quickstart`:

```python
import numpy as np
from tachys.parallel import rank, MASTER

# on every process
state, log_amps, acceptance = sample(nsweeps, state, action, mc_keys, wf)
E_L, e_mean, e2_mean = compute_expectation(H, wf, state, log_amps)

# on the master process only
if rank == MASTER:
    np.save("energy.npy", np.real(e_mean))
```

## One node

A Slurm job is a shell script: the `#SBATCH` lines at its top request the
resources, and the commands that follow run once these are allocated. The
script `job.sh` runs `main.py` on the 4 GPUs of one node:

```bash
#!/bin/bash
#SBATCH --job-name=j1j2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x.%j.out

source $HOME/venvs/tachys/bin/activate    # Python environment with tachys[cuda]

srun python main.py
```

`srun` starts `--nodes` × `--ntasks-per-node` processes, here 4, each running
`python main.py`.

Clusters may require further options, such as `--account`, which names the
project charged for the job, or `--constraint`, which selects a GPU model; their
documentation lists them.

## Several nodes

On several nodes, only `--nodes` changes. For two nodes with 4 GPUs each:

```{code-block} bash
:emphasize-lines: 4

#!/bin/bash
#SBATCH --job-name=j1j2
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x.%j.out

source $HOME/venvs/tachys/bin/activate    # Python environment with tachys[cuda]

srun python main.py
```

`srun` starts 8 processes, 4 on each node, and `N_mc` must be a multiple of 8.
The banner that tachys prints on import, at the top of the output file, gives
the number of GPUs and their distribution over the nodes:

```text
8 × GPU  (NVIDIA H100)
2 nodes  ·  4 GPU/node
```

## Without Slurm

On a machine without Slurm, such as a workstation with several GPUs, `mpirun`
starts the processes; `-np` is their number, one per GPU:

```bash
mpirun -np 2 python main.py
```

JAX recognizes the processes started by Open MPI up to version 4; with other
MPI implementations, Open MPI 5 included, importing tachys fails.
