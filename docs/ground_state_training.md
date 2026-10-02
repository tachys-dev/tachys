# Ground State Training with wandb

The function `train`, in `tachys.ground_state_training`, runs a whole
ground-state optimization: for a given number of steps, it repeats the step of
the loop of the {doc}`quickstart`, which advances the Markov chains, estimates
the energy, and computes and applies the parameter update. It prints the energy
and the timings of every step, and returns the optimized wavefunction. Given a
run of [Weights & Biases](https://wandb.ai) (W&B), a web service that
records the metrics of training runs and plots them, it also logs these
quantities there and saves checkpoints.

## The project

The `wandb` extra of tachys installs W&B, with
[jsonargparse](https://jsonargparse.readthedocs.io), which reads the
configuration file below. This tutorial needs a W&B account, see the
[W&B quickstart](https://docs.coreweave.com/quickstart/).

```bash
pip install "tachys[wandb]"     # tachys[cuda,wandb] on GPUs
wandb login
```

The project consists of four files, and the two directories that the runs
fill:

```text
j1j2_square/
├── vmc_config.yaml    # the parameters
├── config.py          # reads vmc_config.yaml
├── main.py            # builds the model and calls train
├── job.sh             # the Slurm job
├── log_output/        # the output of the jobs
└── wandb/             # the W&B runs
```

`vmc_config.yaml` holds the parameters:

```yaml
seed: 22

wandb:
    log: True
    entity: 'my-team'          # a W&B user or team
    project: 'j1j2_square'
    name: ''                   # appended to the name of the run

checkpoint:
    input: ''                  # checkpoint directory to resume from
    every: 1000
    keep: 1

lattice:
    Lx: 6
    Ly: 6

hamiltonian:
    J1: 1.0
    J2: 0.5

transformer:
    num_layers: 8
    heads: 12
    f: 6                       # features per head: d_model = heads * f
    b: 1                       # patch side

optimizer:
    mode: 'complex'
    diag_shift: 1e-4
    eta0: 0.005
    eta_final: 0.001
    N_steps: 5000

vmc:
    N_mc: 8192
    max_dist: 2
    nsweeps: 1
```

`config.py` reads them into `cfg`, whose fields follow the structure of the
file: `cfg.lattice.Lx`, `cfg.optimizer.N_steps`. An argument on the command
line, such as `--hamiltonian.J2 0.6`, replaces the value of the file.
`config.py` also computes `d_model` and the name of the W&B run.

```python
from jsonargparse import ArgumentParser

parser = ArgumentParser(default_config_files=["vmc_config.yaml"])
parser.add_argument("--seed", type=int)

parser.add_argument("--wandb.log", type=bool)
parser.add_argument("--wandb.entity", type=str)
parser.add_argument("--wandb.project", type=str)
parser.add_argument("--wandb.name", type=str)

parser.add_argument("--checkpoint.input", type=str)
parser.add_argument("--checkpoint.every", type=int)
parser.add_argument("--checkpoint.keep", type=int)

parser.add_argument("--lattice.Lx", type=int)
parser.add_argument("--lattice.Ly", type=int)

parser.add_argument("--hamiltonian.J1", type=float)
parser.add_argument("--hamiltonian.J2", type=float)

parser.add_argument("--transformer.num_layers", type=int)
parser.add_argument("--transformer.heads", type=int)
parser.add_argument("--transformer.f", type=int)
parser.add_argument("--transformer.b", type=int)

parser.add_argument("--optimizer.mode", type=str)
parser.add_argument("--optimizer.diag_shift", type=float)
parser.add_argument("--optimizer.eta0", type=float)
parser.add_argument("--optimizer.eta_final", type=float)
parser.add_argument("--optimizer.N_steps", type=int)

parser.add_argument("--vmc.N_mc", type=int)
parser.add_argument("--vmc.max_dist", type=int)
parser.add_argument("--vmc.nsweeps", type=int)

cfg = parser.parse_args()

cfg.transformer.d_model = cfg.transformer.heads * cfg.transformer.f
cfg.wandb.run_name = (f"{cfg.lattice.Lx}x{cfg.lattice.Ly}_J2={cfg.hamiltonian.J2}"
                      f"_h={cfg.transformer.heads}_f={cfg.transformer.f}")
if cfg.wandb.name:
    cfg.wandb.run_name += f"_{cfg.wandb.name}"
```

## The script

`main.py` sets up the optimization with the parameters in `cfg` and calls
`train`:

```python
from config import cfg

import jax

from tachys.lattice.lattice_database import square
from tachys.lattice.spins.spin_state import SpinState, init_config_fixed_magn
from tachys.lattice.spins.hamiltonians.heisenberg import heisenberg_hamiltonian
from tachys.lattice.spins.sign_rules import MSR_log_phase_square, add_sign_rule
from tachys.lattice.ansatz.spin_vit import SpinViT
from tachys.lattice.bond_exchange import BondExchange
from tachys.wavefunction import WaveFunction
from tachys.optimizer import MARCH, linear_decay
from tachys.ground_state_training import train
from tachys.checkpoint import load_checkpoint, get_last_step
from tachys.parallel import rank, MASTER

key = jax.random.key(cfg.seed)
lattice = square(shape=(cfg.lattice.Lx, cfg.lattice.Ly))
N = lattice.Ns

J1, J2 = cfg.hamiltonian.J1, cfg.hamiltonian.J2
H = heisenberg_hamiltonian(lattice, nn=[
    ((1, 0), J1), ((0, 1), J1),         # nearest neighbours
    ((1, 1), J2), ((1, -1), J2),        # next-nearest neighbours
])

key, subkey = jax.random.split(key)
spins = init_config_fixed_magn(subkey, N, N_mc=cfg.vmc.N_mc)
state = SpinState(spins=spins, lattice=lattice)

b = cfg.transformer.b
model = SpinViT(num_layers=cfg.transformer.num_layers, d_model=cfg.transformer.d_model,
                num_heads=cfg.transformer.heads, seq_len=N // b**2, b=b,
                transl_invariant=True, two_dimensional=True)
key, subkey = jax.random.split(key)
params = model.init(subkey, state)
# Marshall sign rule
apply_fn = add_sign_rule(MSR_log_phase_square, model.apply, lattice.L)
wf = WaveFunction(params=params, apply_fn=apply_fn)

action = BondExchange.create(lattice, max_dist=cfg.vmc.max_dist)
optimizer = MARCH(diag_shift=cfg.optimizer.diag_shift, mode=cfg.optimizer.mode)
lr_schedule = linear_decay(cfg.optimizer.eta0, cfg.optimizer.eta_final, cfg.optimizer.N_steps)

# resume from a checkpoint
opt_state, start_step = None, 0
if cfg.checkpoint.input:
    params, opt_state, state, key = load_checkpoint(
        cfg.checkpoint.input, state, optimizer.init(wf.params), params_template=wf.params,
    )
    wf = wf.replace(params=params)
    start_step = get_last_step(cfg.checkpoint.input)

# the W&B run, on the master process only
wandb_run = None
if cfg.wandb.log and rank == MASTER:
    import wandb
    wandb_run = wandb.init(
        entity=cfg.wandb.entity, project=cfg.wandb.project, name=cfg.wandb.run_name,
        config={**cfg.as_dict(),
                "checkpoint_every": cfg.checkpoint.every,
                "checkpoint_keep": cfg.checkpoint.keep},
    )

key, state, wf, opt_state, history = train(
    key, H, state, wf, optimizer, action, cfg.optimizer.N_steps - start_step,
    lr_schedule, cfg.vmc.N_mc, wandb_run=wandb_run, nsweeps=cfg.vmc.nsweeps,
    opt_state=opt_state, start_step=start_step,
)

if wandb_run is not None:
    wandb_run.finish()
```

The wavefunction is `SpinViT`, whose output is complex, with the Marshall sign
rule ({doc}`guide/wavefunction`). The Monte Carlo move, `BondExchange` with
`max_dist=2`, exchanges the spins of two nearest or next-nearest neighbours
({doc}`guide/sampling`). The optimizer is MARCH, a variant of stochastic
reconfiguration with momentum; the phase of the network depends on the
parameters, so it runs in complex mode ({doc}`guide/optimization`).
`linear_decay` lowers the learning rate from `eta0` to `eta_final` over
`N_steps` steps.

`wandb.init` creates the run in the project `project` of the entity `entity`,
with a name and a config: the parameters of the run, here all of `cfg`, which
W&B shows with the run and uses to filter and group runs. Only the master
process creates the run; on the other processes `wandb_run` is `None`, but
`train` is called by all of them ({doc}`parallel`).

`train` runs `N_steps` steps of the loop of the {doc}`quickstart`, here
`cfg.optimizer.N_steps - start_step`, with `nsweeps` sweeps per step and the
learning rate `lr_schedule(step)` at step `step`. Its other keyword arguments
are

- `wandb_run`, the W&B run, or `None`;
- `opt_state` and `start_step`, the optimizer state and the number of the first
  step, to resume a run ({ref}`training-checkpoints`);
- `log_callback_fn`, a function `(state, wf, step) -> dict` called at every
  step, whose values are logged with the others;
- `skip_optimization`, to sample and measure the energy without updating the
  parameters.

The {doc}`api` describes all of them.

## The output of `train`

A short test on one GPU, an NVIDIA RTX A6000, sets a smaller lattice and
network, fewer chains and fewer steps on the command line:

```bash
python main.py --lattice.Lx 4 --lattice.Ly 4 --transformer.num_layers 2 \
               --vmc.N_mc 512 --optimizer.N_steps 250
```

`train` prints the setup, one line per step, and a summary:

```text
--- Simulation setup ---
Lattice      : Lattice  Ns=16  L=(4, 4)  nb=1  pbc=(True, True)
Hamiltonian  : _OperatorSum  (2 terms)
             : 116,736 parameters
Optimizer    : MARCH(mode=complex, nbatches=1, dtype=None)
MC action    : BondExchange(max_dist=2, bonds=((0, 1), (0, 3), (0, 4), (0, 12), ...), Nbands=1)
N_mc         : 512    N_steps: 250
lr schedule  : 5.00e-03 -> 1.02e-03
------------------------------------------------------------

--- START TRAINING ---
 step │                  E/N │      var/N │   vscore │  accept │        lr │   t_mc │  t_exp │  t_opt │      t_tot │     ETA (h)
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    0 │      -0.085631646915 │   7.99e+00 │ 1089.2874 │   0.278 │  5.00e-03 │   4.06 │   2.20 │   9.30 │  15.68 (s) │    1.08 (h)
    1 │      -0.074122214540 │   1.01e+00 │ 183.8694 │   0.158 │  4.98e-03 │   3.55 │   2.35 │   9.26 │  15.16 (s) │    1.04 (h)
    2 │      -0.088764702597 │   4.73e-01 │  60.0299 │   0.129 │  4.97e-03 │   0.15 │   0.21 │   0.56 │   0.92 (s) │    0.06 (h)
    3 │      -0.095475820411 │   2.67e-01 │  29.3370 │   0.105 │  4.95e-03 │   0.14 │   0.20 │   0.56 │   0.90 (s) │    0.06 (h)
    4 │      -0.096146549477 │   2.27e-01 │  24.6006 │   0.104 │  4.94e-03 │   0.14 │   0.21 │   0.55 │   0.90 (s) │    0.06 (h)
  ...
  248 │      -0.435008411931 │   8.98e-03 │   0.0474 │   0.124 │  1.03e-03 │   0.14 │   0.21 │   0.57 │   0.92 (s) │    0.00 (h)
  249 │      -0.435610718530 │   5.72e-02 │   0.3013 │   0.119 │  1.02e-03 │   0.14 │   0.21 │   0.57 │   1.11 (s) │    0.00 (h)
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Done. 250 steps in 258.4s (1.033s/step avg)
E/N over last 100 steps = -0.428942 ± 5.12e-03  (var = 2.62e-05)
```

| Column | Description |
|---|---|
| `E/N` | The energy per site: the mean of the local energies, divided by $N$. |
| `var/N` | The variance of the local energies, divided by $N$. |
| `vscore` | The V-score $N\sigma^2/E^2$ ({doc}`guide/optimization`). |
| `accept` | The acceptance rate of the moves, averaged over the chains. |
| `lr` | The learning rate. |
| `t_mc`, `t_exp`, `t_opt` | The seconds spent sampling, computing the local energies, and computing and applying the update. |
| `t_tot` | The seconds of the whole step. |
| `ETA (h)` | The hours left, estimated from the duration of the last step. |

The first two steps include the compilation by JAX, so the ETA is meaningful
from the third one. The last line gives the mean of `E/N` over the last 100
steps, or over all of them if there are fewer, with its standard deviation and
variance. If `E/N` becomes NaN or leaves the interval $[-100, 100]$, `train`
stops the program.

`train` returns `(key, state, wf, opt_state, history)`. The first four, the
PRNG key, the last configurations of the chains, the optimized wavefunction and
the optimizer state, are what another call of `train` needs to continue, or
`compute_observables` to measure operators with the optimized wavefunction
({doc}`api`). `history` is a dict of lists, with one entry per step, for the
keys `"energy"` (per site), `"variance_per_site"`, `"vscore"`, `"acceptance"`
and `"lr"`.

## Weights & Biases

At every step, `train` logs the quantities of `history` to the W&B run, under
the same names, against the step. At the end it writes the mean and
the variance of `E/N` over the last 100 steps to the summary of the run, as
`mean_energy` and `var_energy`; a run that diverges gets the tag `divergence`.
Online, `wandb.init` prints the address of the run page, which W&B updates as
the run proceeds. For the test above:

```{figure} images/wandb_charts.png
:alt: The charts of the test run on W&B
:class: screenshot

The test run on W&B: one chart per quantity logged by `train`.
```

The overview of the run lists its config, with the values set on the command
line, and its summary:

```{figure} images/wandb_overview.png
:alt: The overview of the test run on W&B
:class: screenshot

The config and the summary of the test run.
```

### On a cluster

Compute nodes often have no internet access. With the environment variable
`WANDB_MODE=offline`, W&B writes the run to the directory
`wandb/offline-run-<date>_<time>-<id>` only, and `wandb sync` uploads it later
from a machine with internet access, such as the login node. `job.sh` runs the
parameters of `vmc_config.yaml` on the 8 GPUs of one node ({doc}`parallel`):

```{code-block} bash
:emphasize-lines: 11

#!/bin/bash
#SBATCH --job-name=j1j2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --output=log_output/%x.o%j

source $HOME/venvs/tachys/bin/activate    # Python environment with tachys[cuda,wandb]
export WANDB_MODE=offline

srun python main.py
```

The test run above was saved in this way, and uploaded with

```bash
wandb sync wandb/offline-run-20261001_192632-rszl7pcn
```

(training-checkpoints)=

## Checkpoints

With a W&B run, `train` saves checkpoints in the directory of the run,
`wandb_run.dir/checkpoints`, which W&B uploads with the run: one every
`checkpoint_every` steps and one at the last step, keeping the last
`checkpoint_keep`. It reads both from the config of the run, where `main.py`
copies them from the `checkpoint` section of `vmc_config.yaml`. A checkpoint
holds the parameters, the optimizer state, the configurations of the chains
and the PRNG key.

A run that stopped before `N_steps` steps, at the time limit of its job for
instance, resumes from its last checkpoint, with the same parameters and
`checkpoint.input` set to the checkpoint directory:

```bash
python main.py --checkpoint.input wandb/offline-run-<date>_<time>-<id>/files/checkpoints
```

`main.py` restores the state with `load_checkpoint`, and passes the optimizer
state and the number of steps done, `start_step`, to `train`, which runs the
remaining `N_steps - start_step` steps. The learning-rate schedule, the printed
steps and the steps on W&B continue from `start_step`, in a new W&B run.
