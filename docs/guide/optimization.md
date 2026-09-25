# Optimization

An optimization step turns the local energies of the current sample into a
parameter update. The optimizers in `tachys.optimizer` share one interface:

```python
from tachys.optimizer import SR

optimizer = SR(diag_shift=1e-4, mode="complex")
opt_state = optimizer.init(wf.params)

E_L, e_mean, e2_mean = compute_expectation(H, wf, state, log_amps)
updates, opt_state = optimizer(E_L, opt_state, state, wf)
wf = wf.apply_gradients(updates, lr)          # θ ← θ − lr · δθ
```

`opt_state` holds whatever the optimizer carries from one step to the next.

## Stochastic reconfiguration

Gradient descent moves all parameters with the same learning rate, however
much each of them changes the wavefunction. Stochastic reconfiguration (SR)
measures the step by the change of the wavefunction instead. Let $\bar O$ be
the $N_{mc} \times P$ matrix of the log-derivatives
$\partial_{\theta_k} \log\psi(x)$ on the sample, centred and divided by
$\sqrt{N_{mc}}$, and $\bar\varepsilon$ the vector of centred local energies
$E_L(x) - \langle E_L \rangle$, scaled by $2/\sqrt{N_{mc}}$. With the
quantum geometric tensor $S = \bar O^T \bar O$ and the energy gradient
$F = \bar O^T \bar\varepsilon$, the update is

$$
\delta\theta = (S + \lambda I)^{-1} F = \bar O^T \big(\bar O\,\bar O^T + \lambda I\big)^{-1}\,\bar\varepsilon .
$$

tachys evaluates the right-hand side, which inverts the $N_{mc} \times N_{mc}$
neural tangent kernel $\bar O \bar O^T$ instead of the $P \times P$ matrix $S$:
the cost is set by the number of samples, not the number of parameters. This
formulation follows Rende, Viteritti, Bardone, Becca & Goldt, ["A simple linear
algebra identity to optimize large-scale neural network quantum
states"](https://www.nature.com/articles/s42005-024-01732-4), *Communications
Physics* (2024). A similar procedure, with a different regularization, was
developed by Chen & Heyl, ["Empowering deep neural quantum states through
efficient optimization"](https://www.nature.com/articles/s41567-024-02566-1),
*Nature Physics* (2024).

The parameters of `SR` are

- `diag_shift`, the regularization $\lambda$. A larger shift gives smaller,
  more stable steps, and in the limit of large $\lambda$ plain gradient descent
  with learning rate `lr / diag_shift`. The examples use `1e-4` to `1e-3`.
- `mode`, `"complex"` or `"real"`. In `"complex"` mode the real and imaginary
  parts of $\log\psi$ enter $\bar O$ as separate rows, and those of the local
  energies enter $\bar\varepsilon$ in the same way, so that
  $F = 2\,\mathrm{Re}\,\langle \Delta O^{*}\,\Delta E_L\rangle$ is the full energy
  gradient, phase included. `"real"` keeps only $\log|\psi|$: it is exact when
  the phase does not depend on the parameters — a positive wavefunction, a fixed
  sign rule, a Slater determinant of real orbitals — and solves a system half
  the size. In both modes the parameters themselves must be real; a complex
  $\log\psi$ is built from real parameters, as in the built-in ansätze.
- `nbatches`, the number of chunks in which the Jacobian is evaluated, to limit
  memory.
- `dtype`, the precision of the network evaluations inside the update: the
  Jacobian, its contraction into the kernel, and the map back to parameter
  space. The default, `None`, uses the dtype the parameters are stored in.
  `jnp.float32` (or the string `"float32"`) runs them in single precision,
  with full-precision matmuls rather than the TF32 that JAX uses for float32
  on recent NVIDIA GPUs. The kernel, its inversion and the returned update
  stay in double precision. Workstation GPUs have a small fraction of their
  single-precision throughput in double precision, so the saving is largest
  there. Single precision moves the kernel's eigenvalues by about
  $10^{-8}\lambda_{\max}$, with $\lambda_{\max}$ its largest eigenvalue. With a
  smaller `diag_shift`, the shifted kernel can stop being positive definite, and
  the solve fails. A failed solve gives a zero SR
  step, or only the momentum term for SPRING and MARCH. `dtype` is independent
  of `WaveFunction.dtype`, which sets the precision of sampling and of the
  local energies.

## Momentum: SPRING and MARCH

`SPRING` adds momentum to SR, with coefficient `mu`. `MARCH` also rescales each
parameter by a running average, with decay rate `beta`, of the squared change
of its update from one step to the next, in the spirit of Adam's second moment.
SPRING was introduced by Goldshlager, Abrahamsen & Lin, ["A Kaczmarz-inspired
approach to accelerate the optimization of neural network
wavefunctions"](https://doi.org/10.1016/j.jcp.2024.113351), *Journal of
Computational Physics* (2024), and MARCH by Gu et al., ["Solving the Hubbard
model with neural quantum states"](https://doi.org/10.1038/s41467-026-74028-6),
*Nature Communications* (2026).

Both keep their history in `opt_state` and replace `SR` without other changes
to the loop:

```python
from tachys.optimizer import MARCH

optimizer = MARCH(diag_shift=1e-4, mode="complex", mu=0.95, beta=0.995)
opt_state = optimizer.init(wf.params)
```

The values of `mu` and `beta` above are the defaults; for `SPRING`, `mu`
defaults to 0.9.

## Learning-rate schedules

The learning rate is an argument of `apply_gradients`, so it can change at every
step. `linear_decay` and `shifted_cosine_decay` return it as a function of the
step:

```python
from tachys.optimizer import shifted_cosine_decay

N_steps = 30
# cosine decay from 0.03 down to 0.003
lr_schedule = shifted_cosine_decay(init_value=0.03, decay_steps=N_steps)

for step in range(N_steps):
    key, subkey = jax.random.split(key)
    mc_keys = jax.random.split(subkey, N_mc)

    state, log_amps, acceptance = sample(1, state, action, mc_keys, wf)
    E_L, e_mean, e2_mean = compute_expectation(H, wf, state, log_amps)

    updates, opt_state = optimizer(E_L, opt_state, state, wf)
    wf = wf.apply_gradients(updates, lr_schedule(step))
```

## Monitoring convergence

Besides the energy, `compute_expectation` returns $\langle |E_L|^2 \rangle$,
and with it the variance of the energy,
$\sigma^2 = \langle |E_L|^2 \rangle - |\langle E_L \rangle|^2$, which vanishes
for an exact eigenstate. It measures convergence without reference to the
unknown exact energy; its dimensionless form, the V-score
$N_s\,\sigma^2 / \langle E_L \rangle^2$, compares different systems.

## The `train` driver

`train` (`tachys.ground_state_training`) runs the same loop and prints, at
every step, the energy per site, the variance, the V-score, the acceptance and
the timings:

```python
from tachys.ground_state_training import train
from tachys.optimizer import shifted_cosine_decay

N_steps = 30
lr_schedule = shifted_cosine_decay(init_value=0.03, decay_steps=N_steps)
key, state, wf, opt_state, history = train(
    key, H, state, wf, optimizer, action, N_steps, lr_schedule, N_mc,
)
```

`history` holds the same quantities, one entry per step. Given a Weights &
Biases run as `wandb_run`, `train` logs them there and writes checkpoints. To
resume, restore the checkpoint with `tachys.checkpoint.load_checkpoint` and
pass the optimizer state and the step reached as `opt_state` and `start_step`.
