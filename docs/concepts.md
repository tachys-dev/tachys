---
html_theme.sidebar_secondary.remove: true
---

# Core Concepts

Variational Monte Carlo (VMC) approximates the ground state of a Hamiltonian
$H$ by a wavefunction $\psi_\theta$ whose parameters $\theta$ minimize the
energy

$$
E(\theta) = \frac{\langle \psi_\theta | H | \psi_\theta \rangle}{\langle \psi_\theta | \psi_\theta \rangle} \;\geq\; E_0 .
$$

tachys exposes the basic operations of Monte Carlo as separate functions:
sampling, local estimators and parameter updates. A VMC run is a plain Python
loop that calls them, one per step of the algorithm, and the same functions are
the building blocks of {ref}`other algorithms <vmc-reuse>`. They take immutable
JAX pytrees and return new ones; they are compiled with `jit` and run in
parallel on all available devices.

:::{list-table}
:header-rows: 1
:widths: auto
:class: vmc-algo

* -
  - Variational Monte Carlo
  - vmc.py
* -
  - Wavefunction

    $\psi_\theta$
  - ```{code-block} python
    :linenos:
    :lineno-start: 1
    wf = WaveFunction(params=params, apply_fn=model.apply)
    ```
* -
  - Configurations

    $x_1, \dots, x_{N_{mc}}$
  - ```{code-block} python
    :linenos:
    :lineno-start: 2
    state = SpinState(spins=spins, lattice=lattice)
    ```
* -
  - **for** each step
  - ```{code-block} python
    :linenos:
    :lineno-start: 3
    for step in range(N_steps):
    ```
* - {ref}`1 <vmc-sampling>`
  - Sampling

    $x_i \sim |\psi_\theta(x)|^2$
  - ```{code-block} python
    :linenos:
    :lineno-start: 4
        key, subkey = jax.random.split(key)
        mc_keys = jax.random.split(subkey, N_mc)
        state, log_amps, acceptance = sample(N_sweeps, state, action, mc_keys, wf)
    ```
* - {ref}`2 <vmc-energy>`
  - Local energies

    $E_L(x_i) = \dfrac{\langle x_i | H | \psi_\theta \rangle}{\langle x_i | \psi_\theta \rangle}$
  - ```{code-block} python
    :linenos:
    :lineno-start: 7
        E_L, e_mean, e2_mean = compute_expectation(H, wf, state, log_amps)
    ```
* - {ref}`3 <vmc-direction>`
  - Update direction

    $\delta\theta = (S + \lambda I)^{-1} F$
  - ```{code-block} python
    :linenos:
    :lineno-start: 8
        updates, opt_state = optimizer(E_L, opt_state, state, wf)
    ```
* - {ref}`4 <vmc-update>`
  - Parameter update

    $\theta \leftarrow \theta - \eta\, \delta\theta$
  - ```{code-block} python
    :linenos:
    :lineno-start: 9
        wf = wf.apply_gradients(updates, lr)
    ```
:::

Each iteration rebinds `state`, `wf` and `opt_state` to new values, like the
carry of a `jax.lax.scan`; nothing is stored anywhere else. The
{doc}`quickstart` gives runnable versions.

## The variational state

A configuration $x$ holds the spins, or the occupation numbers, of the $N_s$
sites. The network maps it to a complex log-amplitude $\log\psi_\theta(x)$:
the real part is $\log|\psi_\theta(x)|$, the imaginary part the phase.

A `State` holds the $N_{mc}$ configurations in an array whose leading axis runs
over the chains, with the `Lattice` ({doc}`guide/lattices`) as static
metadata. A `WaveFunction` pairs the parameters $\theta$ with the `apply`
function of the network ({doc}`guide/wavefunctions`):

```python
log_psi = wf.apply_fn(wf.params, state)      # log ψ_θ(x_i), shape (N_mc,)
```

Both are immutable: `replace` returns a new object and leaves the original
unchanged.

```python
flipped = state.replace(spins=-state.spins)  # state is unchanged
```

(vmc-sampling)=
## 1. Sampling

The energy is an average over $p_\theta(x) \propto |\psi_\theta(x)|^2$, which
the Metropolis–Hastings algorithm samples. A move proposes $x'$ with
probability $q(x' \mid x)$, and the chain accepts it with probability

$$
A(x \to x') = \min\left(1,\; \frac{|\psi_\theta(x')|^2}{|\psi_\theta(x)|^2}\,\frac{q(x \mid x')}{q(x' \mid x)}\right).
$$

The move is the `action`. `sample` advances every chain by `N_sweeps` sweeps,
with one PRNG key per chain, and returns the new configurations, their
log-amplitudes and the acceptance rate ({doc}`guide/sampling`):

```python
action = BondExchange.create(lattice)        # proposes x' from x
state, log_amps, acceptance = sample(N_sweeps, state, action, mc_keys, wf)
```

(vmc-energy)=
## 2. Local energies

The energy is the mean of the local energy over the sample,

$$
E_L(x) = \sum_{x'} \langle x | H | x' \rangle \, \frac{\psi_\theta(x')}{\psi_\theta(x)} ,
\qquad
E(\theta) \approx \frac{1}{N_{mc}} \sum_{i=1}^{N_{mc}} E_L(x_i) .
$$

`compute_expectation` returns the local energies, their mean, and the mean of
$|E_L|^2$. The variance
$\sigma^2 = \langle |E_L|^2 \rangle - |\langle E_L \rangle|^2$ vanishes for an
eigenstate. Any operator can take the place of `H` ({doc}`guide/hamiltonians`).

```python
E_L, e_mean, e2_mean = compute_expectation(H, wf, state, log_amps)
variance = e2_mean - abs(e_mean) ** 2
```

(vmc-direction)=
## 3. Update direction

With the log-derivatives $O_k(x) = \partial_{\theta_k} \log\psi_\theta(x)$,
and $\Delta A = A - \langle A \rangle$ for any quantity $A$, the energy
gradient and the quantum geometric tensor are

$$
F_k = 2\,\mathrm{Re}\,\langle \Delta O_k^{*} \, \Delta E_L \rangle ,
\qquad
S_{kl} = \mathrm{Re}\,\langle \Delta O_k^{*} \, \Delta O_l \rangle .
$$

Stochastic reconfiguration (SR) preconditions the gradient with $S$,
regularized by the diagonal shift $\lambda$:
$\delta\theta = (S + \lambda I)^{-1} F$. The optimizer returns $\delta\theta$
as `updates`, together with its new state, which carries what it keeps from
one step to the next, such as the momentum of `SPRING` and `MARCH`
({doc}`guide/optimization`):

```python
optimizer = SR(diag_shift=1e-4, mode="complex")   # λ = 1e-4
opt_state = optimizer.init(wf.params)
updates, opt_state = optimizer(E_L, opt_state, state, wf)
```

(vmc-update)=
## 4. Parameter update

`apply_gradients` returns a new `WaveFunction` with the parameters
$\theta - \eta\,\delta\theta$. The learning rate $\eta$, `lr`, can change at
every step ({doc}`guide/optimization`).

```python
wf = wf.apply_gradients(updates, lr)
```

(vmc-reuse)=
## Other algorithms

Any step of the loop can be replaced, and the functions combined in other ways,
to implement other algorithms. Real-time evolution (t-VMC) keeps the first two
steps and changes the last two: `TDVP` returns the time derivative
$\dot\theta$ of the parameters, with the call of an optimizer, and the
parameters move forward in time instead of down the gradient:

```{code-block} python
:emphasize-lines: 9, 10

from tachys.dynamics import TDVP

tdvp = TDVP(mode="complex")
opt_state = tdvp.init(wf.params)
dt = 0.01

state, log_amps, acceptance = sample(N_sweeps, state, action, mc_keys, wf)
E_L, e_mean, e2_mean = compute_expectation(H, wf, state, log_amps)
dtheta_dt, opt_state = tdvp(E_L, opt_state, state, wf)
wf = wf.replace(params=jax.tree.map(lambda p, v: p + dt * v, wf.params, dtheta_dt))
```

This is an Euler step; `evolve` (see the {doc}`api`) makes the same calls at
each stage of a Runge–Kutta integrator. In the same way, `compute_expectation`
measures any observable on the sample, any optimizer with the interface of `SR`
can take its place, and {doc}`foundation_models` trains one network on many
Hamiltonians with the loop unchanged.
