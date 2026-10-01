# Pfaffian Wave Function for the Kondo-Heisenberg Model

The Kondo-Heisenberg model describes heavy-fermion materials: a band of
conduction electrons, exchange-coupled on every site to a localized spin. At
weak Kondo coupling $J_K$, the localized spins order antiferromagnetically, and
the Fermi surface encloses the conduction electrons alone; at strong coupling,
the conduction electrons screen them, and the spins join the Fermi sea of a
heavy Fermi liquid. Away from half filling, quantum Monte Carlo has a sign
problem for this model.
[Nikolaenko, Rende, Viteritti, Sachdev & Zhang (2026)](https://arxiv.org/abs/2606.23799)
describe the antiferromagnet, the heavy Fermi liquid and the $d$-wave
superconductor found between them with a single variational wave function, a
transformer with a Pfaffian output layer: the state of lowest energy emerges
from the optimization, without being built into the ansatz.

In this tutorial, we show how to find the ground state of the
Kondo-Heisenberg model,

$$
H = -t \sum_{\langle i,j \rangle, \sigma} \big(c^\dagger_{i\sigma} c_{j\sigma} + \text{h.c.}\big)
    + J \sum_{\langle i,j \rangle} \mathbf{S}_i \cdot \mathbf{S}_j
    + J_K \sum_i \mathbf{S}_i \cdot \mathbf{s}_i ,
$$

on an $L \times L$ square lattice with periodic boundaries, with the wave
function of the paper. Here $c^\dagger_{i\sigma}$ creates a conduction electron
with spin $\sigma$ on site $i$,
$\mathbf{s}_i = \frac12 \sum_{\sigma\sigma'} c^\dagger_{i\sigma} \boldsymbol{\tau}_{\sigma\sigma'} c_{i\sigma'}$
is the spin of the conduction electrons on the site, with $\boldsymbol{\tau}$
the Pauli matrices, and $\mathbf{S}_i$ is the local moment, a spin-½. As in the
paper, we set $t = 1$ and $J = 0.5$, with $N_e = (1 - \delta)\,N$ conduction
electrons on the $N = L^2$ sites, at doping $\delta = 1/4$.

## Configurations

Each site carries a conduction-electron orbital, empty, singly or doubly
occupied, and a local moment, so its configuration
$(n_{i\uparrow}, n_{i\downarrow}, S^z_i)$ takes 8 values. In tachys, the
configurations of the Markov chains are stored in a `State`: a `SpinState` for
spins, a `FermionState` for fermions ({doc}`../guide/configuration`). We show
how to define a custom state that holds both degrees of freedom.

```python
import jax
import jax.numpy as jnp
from flax import struct
from tachys.lattice.state import State


class KondoState(State):
    occupations: jax.Array        # (batch, 2N): n_{i↑}, then n_{i↓}, as in FermionState
    spins: jax.Array              # (batch, N): the moments, ±1, as in SpinState
    Ne: int = struct.field(pytree_node=False)

    @property
    def array(self):              # (batch, 3N): the occupations, then the moments
        return jnp.concatenate((self.occupations, self.spins), axis=-1)

    def replace_array(self, array):
        n = self.occupations.shape[-1]
        return self.replace(occupations=array[..., :n].astype(self.occupations.dtype),
                            spins=array[..., n:].astype(self.spins.dtype))
```

We draw the initial configurations of the chains with the helpers of tachys:
`init_config_spinful` for the electrons, with $N_e/2$ of each spin, and
`init_config_fixed_magn` for the moments, with zero magnetization. The state
also holds the lattice, here the $L \times L$ square lattice with periodic
boundaries, `square(shape=(L, L))` ({doc}`../guide/lattice`):

```python
import jax
from tachys.lattice.lattice_database import square
from tachys.lattice.fermions.fermion_state import init_config_spinful
from tachys.lattice.spins.spin_state import init_config_fixed_magn

L, Ne, N_mc = 8, 48, 200
key_n, key_s = jax.random.split(jax.random.key(0))
occupations, N_up, N_down = init_config_spinful(key_n, Ns=L * L, Ne=Ne, N_mc=N_mc)
spins = init_config_fixed_magn(key_s, L * L, N_mc=N_mc)
state = KondoState(occupations=occupations, spins=spins, Ne=Ne,
                   lattice=square(shape=(L, L)))
state.array.shape                       # (200, 192)
```

## Hamiltonian

In tachys, a Hamiltonian is a sum of products of elementary operators, written
like the symbols of the formula: `Cup_dag(i) * Cup(j)` is
$c^\dagger_{i\uparrow} c_{j\uparrow}$, and `Splus(i) * Sminus(j)` is
$S^+_i S^-_j$ ({doc}`../guide/hamiltonian`). For the Kondo term, we use

$$
\mathbf{S}_i \cdot \mathbf{s}_i = S^z_i\, s^z_i
+ \tfrac12 \big( S^+_i\, c^\dagger_{i\downarrow} c_{i\uparrow} + S^-_i\, c^\dagger_{i\uparrow} c_{i\downarrow} \big) ,
$$

with $s^z_i = \frac12 (n_{i\uparrow} - n_{i\downarrow})$, which is `Sz_f(i)`:

```python
from tachys.lattice.spins.spin_operators import Sz, Splus, Sminus
from tachys.lattice.fermions.fermion_operators import (
    Cup, Cup_dag, Cdn, Cdn_dag, Sz_f,
)


def kondo_heisenberg_hamiltonian(L, t, J, JK):
    """The Kondo-Heisenberg model on an L × L square lattice, periodic boundaries."""
    H = None
    for x in range(L):
        for y in range(L):
            i     = x + L * y               # site (x, y)
            right = (x + 1) % L + L * y     # site (x + 1, y)
            up    = x + L * ((y + 1) % L)   # site (x, y + 1)
            for j in [right, up]:
                term = ((-t) * Cup_dag(i) * Cup(j) + (-t) * Cup_dag(j) * Cup(i)
                        + (-t) * Cdn_dag(i) * Cdn(j) + (-t) * Cdn_dag(j) * Cdn(i)
                        + J * Sz(i) * Sz(j)
                        + (J / 2) * Splus(i) * Sminus(j)
                        + (J / 2) * Sminus(i) * Splus(j))
                H = term if H is None else H + term
            H = H + (JK * Sz(i) * Sz_f(i)
                     + (JK / 2) * Splus(i) * Cdn_dag(i) * Cup(i)
                     + (JK / 2) * Sminus(i) * Cup_dag(i) * Cdn(i))
    return H
```

## Monte Carlo moves

tachys samples the configurations from $|\Psi|^2$ with the Metropolis–Hastings
algorithm: at each step, a move proposes a new configuration on every chain
({doc}`../guide/sampling`). The built-in move `BondExchange` swaps the values
of two neighbouring sites within one band of the state's `array`. On the three
bands of a `KondoState`, `BondExchange.create(lattice, Nbands=3)` moves an
electron to a neighbouring site, keeping its spin, or exchanges two
antiparallel moments. These moves conserve the number of electrons of each
spin and the magnetization of the moments, whereas the Kondo term flips the
spin of an electron together with the moment on its site.
We define a move for this process. A move is a subclass of `_BaseAction`,
whose `__call__` receives one random key per chain and the configurations, and
returns the proposed configurations, whether each proposal is allowed, and the
logarithm of the ratio of the proposal probabilities. `KondoFlip` draws a
site, and flips both spins if the site is singly occupied and its moment is
antiparallel to the electron.

```python
import jax
import jax.numpy as jnp
from tachys.montecarlo import _BaseAction


class KondoFlip(_BaseAction):
    def __call__(self, key, state):
        n, S = state.occupations, state.spins        # (N_mc, 2N), (N_mc, N)
        chains = jnp.arange(S.shape[0])
        N = S.shape[1]
        i = jax.vmap(lambda k: jax.random.randint(k, (), 0, N))(key)   # one site per chain
        n_up, n_dn, S_i = n[chains, i], n[chains, i + N], S[chains, i]
        allowed = (n_up + n_dn == 1) & (S_i == n_dn - n_up)   # e.g. electron ↑, moment ↓
        new_n = n.at[chains, i].set(n_dn).at[chains, i + N].set(n_up)
        new_S = S.at[chains, i].set(-S_i)
        return state.replace(occupations=new_n, spins=new_S), allowed, 0.0
```

The site is drawn uniformly, so the proposal is symmetric and the logarithm is
zero. In the optimization below, a `CompositeAction` chooses one of the two
moves on each chain at each step, with probabilities 0.8 and 0.2.

## The Pfaffian wave function

We use the wave function of the paper, a Pfaffian over the modes of the
conduction electrons occupied in the configuration $s$,

$$
\Psi(s) = \mathrm{Pf}\big[ \phi_R(s)\, A\, \phi_R(s)^T \big] ,
$$

where $\phi(s)$ is a $2N \times 2N$ matrix of orbitals, $\phi_R(s)$ the
$N_e \times 2N$ matrix of its rows of the occupied modes $R$, and $A$ a
trainable antisymmetric $2N \times 2N$ matrix. With orbitals that do not
depend on $s$, $\Psi$ is the amplitude of the pair state

$$
|\Psi\rangle = \Big( \sum_{r < r'} F_{rr'}\, c^\dagger_r c^\dagger_{r'} \Big)^{N_e/2} |0\rangle ,
\qquad F = \phi A \phi^T ,
$$

where $r$ and $r'$ run over the $2N$ modes: the ground state of the most
general quadratic Hamiltonian, with pairing and magnetic order, projected on
$N_e$ electrons, whereas a Slater determinant has no pairing. The pairs join
electrons of any spins, so $\Psi$ also describes the configurations with
$N_\uparrow \neq N_\downarrow$ that the Kondo term creates.

Augmented by a neural network, a Pfaffian becomes an accurate variational
wave function for interacting fermions
([Chen et al., 2026](https://doi.org/10.1073/pnas.2535288123)). As in the
paper, we take the orbitals as a backflow,

$$
\phi_{(i,\sigma),\alpha}(s) = \sum_{\beta=1}^{d} y_{i\beta}(s)\, W_{i\sigma\alpha\beta} ,
$$

where $\mathbf{y}_i(s) \in \mathbb{R}^d$ is the output of a transformer for
site $i$, and $W$ a tensor of $N \times 2 \times 2N \times d$ parameters. The
Pfaffian runs over the conduction electrons only, and the moments enter
through the backflow: the transformer reads the configuration of every site,
moments included, so the orbitals $\phi(s)$, and with them the pairing of the
electrons, depend on the configuration of the moments.

## Architecture

We write the network as a Flax module, in three steps from a configuration $s$
to $\log\Psi(s)$:

1. **Embedding.** A lookup table maps the configuration
   $(n_{i\uparrow}, n_{i\downarrow}, S^z_i)$ of each site, one of 8, to a
   vector $\mathbf{x}_i \in \mathbb{R}^d$, which gives a sequence of $N$
   vectors, one per site
   ([Rende et al., 2026](https://arxiv.org/abs/2603.02316)).
2. **Transformer encoder.** A stack of $n_l$ layers with spatial attention and
   real parameters maps the sequence to the backflow vectors
   $(\mathbf{y}_1, \dots, \mathbf{y}_N)$, followed by a layer normalization.
3. **Pfaffian layer.** The orbitals $\phi(s)$, the pairing matrix of the
   occupied modes, and its Pfaffian.

## 1. Spatial attention

We use the encoder of the {doc}`vit_wavefunction`, with one vector for each
site. The spatial attention combines the $N$ vectors as

$$
\mathbf{A}_i = \sum_{j=1}^{N}
\frac{e^{-\gamma d(i,j)}}{\sum_{j'=1}^{N} e^{-\gamma d(i,j')}}\,
\alpha_{i-j}\, V\mathbf{x}_j ,
$$

with trainable weights $\alpha_{i-j}$, which depend only on the displacement
between the two sites, a trainable inverse length $\gamma$, equal to 2 at
initialization, and $d(i,j)$ the distance between the two sites, with the
minimum image.

In the code, we label a displacement like a site: the displacement
$\mathbf{r}_i - \mathbf{r}_j = (dx, dy)$, modulo $L$, gets the index
`k = dx + L * dy` of the site in column $dx$ and row $dy$
({doc}`../guide/lattice`). `displacements(L)` returns `k[i, j]` for every pair
of sites and the length `d[k]` of every displacement. Each head stores one
weight per displacement, `alpha[head, k]`; indexing with `k` spreads them over
all pairs of sites.

```python
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from einops import rearrange


def displacements(L):
    """k[i, j]: index of the displacement r_i - r_j, labelled like the sites;
    d[k]: length of displacement k, with the minimum image."""
    x, y = np.arange(L * L) % L, np.arange(L * L) // L     # site i = x + L * y
    k = (x[:, None] - x) % L + L * ((y[:, None] - y) % L)
    d = np.hypot(np.minimum(x, L - x), np.minimum(y, L - y))   # k = (x_k, y_k)
    return k, d


class SpatialAttention(nn.Module):
    d_model: int
    n_heads: int
    L: int

    @nn.compact
    def __call__(self, x):                        # x: (batch, N, d_model), N = L * L
        N = self.L ** 2
        k, d = displacements(self.L)
        alpha = self.param("alpha", nn.initializers.xavier_uniform(),
                           (self.n_heads, N), jnp.float64)        # (head, k)
        gamma = self.param("gamma", nn.initializers.constant(2.0),
                           (self.n_heads, 1), jnp.float64)
        decay = jax.nn.softmax(-gamma * d, axis=-1)     # e^{-γ d_k} / Σ_k' e^{-γ d_k'}
        w = (decay * alpha)[:, k]                       # w[head, i, j]

        V = nn.Dense(self.d_model, kernel_init=nn.initializers.xavier_uniform(),
                     param_dtype=jnp.float64)
        out = nn.Dense(self.d_model, kernel_init=nn.initializers.xavier_uniform(),
                       param_dtype=jnp.float64)
        v = rearrange(V(x), "b j (h e) -> b h j e", h=self.n_heads)
        A = w @ v                                       # A_i = Σ_j w_ij V x_j
        return out(rearrange(A, "b h i e -> b i (h e)"))
```

An encoder block applies the attention, then a two-layer feed-forward network,
each after a layer normalization and with a skip connection:

```python
import jax.numpy as jnp
import flax.linen as nn


class EncoderBlock(nn.Module):
    d_model: int
    n_heads: int
    L: int

    @nn.compact
    def __call__(self, x):
        y = nn.LayerNorm(param_dtype=jnp.float64)(x)
        x = x + SpatialAttention(self.d_model, self.n_heads, self.L)(y)

        y = nn.LayerNorm(param_dtype=jnp.float64)(x)
        y = nn.Dense(4 * self.d_model, kernel_init=nn.initializers.xavier_uniform(),
                     param_dtype=jnp.float64)(y)
        y = nn.Dense(self.d_model, kernel_init=nn.initializers.xavier_uniform(),
                     param_dtype=jnp.float64)(nn.gelu(y))
        return x + y
```

## 2. Pfaffian layer

The Pfaffian layer computes the orbitals $\phi(s)$ from the backflow vectors,
keeps the rows of the occupied modes, and returns
$\log\mathrm{Pf}\big[\phi_R A \phi_R^T\big]$, with $A$ stored as its
$N(2N - 1)$ entries above the diagonal. We compute the Pfaffians with lrux
([Chen & Roth, 2026](https://arxiv.org/abs/2602.05255); `pip install lrux`), a
JAX library of determinants and Pfaffians and of their low-rank updates: its
`slogpf` returns the sign of the Pfaffian and the logarithm of its absolute
value, as `jnp.linalg.slogdet` does for a determinant.

In some regions of the phase diagram, the optimization can
remain in metastable stripe states, which the paper avoids by seeding the
pairing. With `add_dwave_bias=True`, the layer adds to the pairing matrix a
configuration-independent singlet $d_{x^2-y^2}$ pairing between neighbouring
sites,
$(\Phi_d)_{(i\uparrow),(j\downarrow)} = -(\Phi_d)_{(j\downarrow),(i\uparrow)} = \Delta_d\, h_{i-j}$,
with $h_{i-j} = +1$ for $\mathbf{r}_i - \mathbf{r}_j = \pm\hat{x}$, $-1$ for
$\pm\hat{y}$ and zero otherwise, and a single trainable amplitude $\Delta_d$.

```python
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from lrux import slogpf


def dwave_form_factor(L):
    """h[k]: +1 for the displacements ±x, -1 for ±y, 0 otherwise."""
    h = np.zeros(L * L)
    h[[1, L - 1]] = 1.0                           # (±1, 0)
    h[[L, L * (L - 1)]] = -1.0                    # (0, ±1)
    return h


class PfaffianHead(nn.Module):
    d_model: int
    L: int
    Ne: int
    add_dwave_bias: bool = False

    @nn.compact
    def __call__(self, y, occupations):          # y: (batch, L * L, d_model)
        N = self.L ** 2
        W = self.param("W", nn.initializers.xavier_uniform(),
                       (2, N, self.d_model, 2 * N), jnp.float64)   # W[σ, i, β, α]
        a = self.param("A", nn.initializers.xavier_uniform(),
                       (1, N * (2 * N - 1)), jnp.float64)          # A above the diagonal
        rows, cols = np.triu_indices(2 * N, k=1)
        A = jnp.zeros((2 * N, 2 * N), a.dtype).at[rows, cols].set(a[0])
        A = A - A.T

        phi = jnp.einsum("bid,sida->bsia", y, W).reshape(-1, 2 * N, 2 * N)  # φ[(σ, i), α]
        R = jax.vmap(lambda n: jnp.nonzero(n, size=self.Ne)[0])(occupations)
        phi_R = jnp.take_along_axis(phi, R[:, :, None], axis=1)   # (batch, Ne, 2N)
        F = phi_R @ A @ phi_R.transpose(0, 2, 1)                  # (batch, Ne, Ne)

        if self.add_dwave_bias:
            k, _ = displacements(self.L)
            delta_d = self.param("delta_d", nn.initializers.constant(1.0),
                                 (1,), jnp.float64)
            D = delta_d * dwave_form_factor(self.L)[k]            # Δ_d h_{i-j}
            zero = jnp.zeros((N, N))
            Phi_d = jnp.block([[zero, D], [-D, zero]])            # modes (i, ↑), (j, ↓)
            F = F + Phi_d[R[:, :, None], R[:, None, :]]

        sign, log_abs = slogpf(F)
        return log_abs + 1j * jnp.pi * (sign < 0)                 # log Pf: (batch,)
```

## 3. The wave function

`KondoPfaffian` chains the three steps: it takes a `KondoState` holding a batch
of configurations and returns one log-amplitude per configuration
({doc}`../guide/wavefunction`). The configuration of a site is labelled by
$n_{i\uparrow} + 2\,n_{i\downarrow} + 2\,(S^z_i + 1)$, from 0 to 7, with the
moment stored as $\pm 1$, and `MSR_log_phase_square`, from tachys, adds the
Marshall sign of the moments.

```python
import jax.numpy as jnp
import flax.linen as nn
from tachys.lattice.spins.sign_rules import MSR_log_phase_square


class KondoPfaffian(nn.Module):
    num_layers: int
    d_model: int
    n_heads: int
    add_dwave_bias: bool = False

    @nn.compact
    def __call__(self, state):
        n, S = state.occupations, state.spins     # (batch, 2 * L * L), (batch, L * L)
        N = S.shape[-1]
        L = round(N ** 0.5)
        tokens = n[:, :N] + 2 * n[:, N:] + 2 * (S + 1)             # 0, ..., 7
        x = nn.Embed(8, self.d_model, embedding_init=nn.initializers.xavier_uniform(),
                     param_dtype=jnp.float64)(tokens)              # (batch, N, d_model)
        for _ in range(self.num_layers):
            x = EncoderBlock(self.d_model, self.n_heads, L)(x)
        y = nn.LayerNorm(param_dtype=jnp.float64)(x)
        log_pf = PfaffianHead(self.d_model, L, state.Ne, self.add_dwave_bias)(y, n)
        return log_pf + MSR_log_phase_square(S, L)
```

```python
import jax
from tachys.lattice.lattice_database import square
from tachys.lattice.fermions.fermion_state import init_config_spinful
from tachys.lattice.spins.spin_state import init_config_fixed_magn

L, Ne, N_mc = 8, 48, 200
key_n, key_s = jax.random.split(jax.random.key(0))
occupations, N_up, N_down = init_config_spinful(key_n, Ns=L * L, Ne=Ne, N_mc=N_mc)
spins = init_config_fixed_magn(key_s, L * L, N_mc=N_mc)
state = KondoState(occupations=occupations, spins=spins, Ne=Ne,
                   lattice=square(shape=(L, L)))

model = KondoPfaffian(num_layers=4, d_model=72, n_heads=12)
params = model.init(jax.random.key(1), state)
model.apply(params, state).shape        # (200,), complex128
```

## Ground-state optimization

We optimize the wave function with the settings of the paper, at $J_K = 2$ on
an $8 \times 8$ lattice with $N_e = 48$ electrons. The script follows the
{doc}`../quickstart`: `WaveFunction` pairs the parameters with the network, and
at each step `sample` advances the Markov chains, `compute_expectation` returns
the local energies and their mean, and the optimizer turns the local energies
into a parameter update. The network has $n_l = 4$ layers, $h = 12$ heads and
$d = 72$, for 1,402,145 parameters, most of them in $W$. The optimizer is
MARCH, a variant of stochastic reconfiguration with momentum, in real mode,
since the Pfaffian is real and the Marshall sign is fixed
({doc}`../guide/optimization`). It takes 20,000 steps of 8192 samples, with the
learning rate of `shifted_cosine_decay`, which decays from 0.005 to 0.005/3
along a cosine. $J_K = 2$ lies in the superconducting region, so we start from
the $d$-wave seed.

```python
import jax
import jax.numpy as jnp

from tachys.lattice.lattice_database import square
from tachys.lattice.fermions.fermion_state import init_config_spinful
from tachys.lattice.spins.spin_state import init_config_fixed_magn
from tachys.lattice.bond_exchange import BondExchange
from tachys.montecarlo import CompositeAction, sample
from tachys.wavefunction import WaveFunction
from tachys.optimizer import MARCH, shifted_cosine_decay
from tachys.lattice.operator.local_estimator import compute_expectation

L = 8
lattice = square(shape=(L, L))
N = lattice.Ns
Ne = 48                                 # doping δ = 1 - Ne / N = 1/4

H = kondo_heisenberg_hamiltonian(L, t=1.0, J=0.5, JK=2.0)

key = jax.random.key(0)
N_mc = 8192
key, key_n, key_s = jax.random.split(key, 3)
occupations, N_up, N_down = init_config_spinful(key_n, Ns=N, Ne=Ne, N_mc=N_mc)
spins = init_config_fixed_magn(key_s, N, N_mc=N_mc)
state = KondoState(occupations=occupations, spins=spins, Ne=Ne, lattice=lattice)

model = KondoPfaffian(num_layers=4, d_model=72, n_heads=12, add_dwave_bias=True)
key, subkey = jax.random.split(key)
params = model.init(subkey, state)
wf = WaveFunction(params=params, apply_fn=model.apply)
print(wf.num_params)                    # 1402145

action = CompositeAction(
    actions=(BondExchange.create(lattice, Nbands=3), KondoFlip()),
    probs=(0.8, 0.2),
)
optimizer = MARCH(diag_shift=1e-4, mode="real")
opt_state = optimizer.init(wf.params)

N_steps = 20000
lr_schedule = shifted_cosine_decay(0.005, N_steps, min_value=0.005 / 3)
for step in range(N_steps):
    key, subkey = jax.random.split(key)
    mc_keys = jax.random.split(subkey, N_mc)

    state, log_amps, acceptance = sample(1, state, action, mc_keys, wf)
    E_L, e_mean, e2_mean = compute_expectation(H, wf, state, log_amps)

    updates, opt_state = optimizer(E_L, opt_state, state, wf)
    wf = wf.apply_gradients(updates, lr_schedule(step))

    if step % 100 == 0:
        print(f"step {step:5d}  E/N = {jnp.real(e_mean) / N: .5f}")
```

At this size, the optimization runs on several GPUs, with one process per GPU
({doc}`../parallel`). The paper, on NVIDIA H200 GPUs, reaches
$E/N = -2.08019(2)$ at $J_K = 2$, with a variance per site of $0.012(1)$.

## Structure factor and momentum distribution

We show how to measure the structure factor of the moments and the momentum
distribution of the conduction electrons, which identify the phases of the
paper,

$$
S(\mathbf{k}) = \frac1N \sum_{i,j} e^{i\mathbf{k}\cdot(\mathbf{r}_i - \mathbf{r}_j)}
\langle \mathbf{S}_i \cdot \mathbf{S}_j \rangle ,
\qquad
n(\mathbf{k}) = \frac12 \sum_\sigma \langle c^\dagger_{\mathbf{k}\sigma} c_{\mathbf{k}\sigma} \rangle .
$$

Both are Fourier transforms of correlations averaged over the translations,
$C(\mathbf{r}) = \frac1N \sum_i \langle \mathbf{S}_i \cdot \mathbf{S}_{i+\mathbf{r}} \rangle$
and
$G(\mathbf{r}) = \frac1{2N} \sum_{i,\sigma} \langle c^\dagger_{i\sigma} c_{i+\mathbf{r},\sigma} \rangle$.
In the antiferromagnet, $S(\mathbf{k})$ peaks at $(\pi, \pi)$, with
$m_s^2 = S(\pi, \pi)/N$ the squared staggered magnetization; the Fermi surface,
where $n(\mathbf{k})$ drops, encloses an area that counts the conduction
electrons alone in the antiferromagnet, and the conduction electrons and the
moments in the heavy Fermi liquid. The pair correlations of the paper involve
only the conduction electrons, and are measured as in the
{doc}`sbp_wavefunction`.

We average each operator over the $N$ positions of the origin with arrays of
sites ({doc}`../guide/hamiltonian`), where `shift(dx, dy)` gives the site
$\mathbf{r}_i + (dx, dy)$ for every site $i$. `compute_observables` runs 2
sampling steps with the optimized wave function, and returns, for each list of
operators, the mean of every operator at each step.

```python
import numpy as np
from tachys.lattice.spins.spin_operators import Sz, Splus, Sminus
from tachys.lattice.fermions.fermion_operators import Cup, Cup_dag, Cdn, Cdn_dag
from tachys.ground_state_training import compute_observables

x, y = np.arange(N) % L, np.arange(N) // L      # site i = x + L * y

def shift(dx, dy):
    """The site r_i + (dx, dy), for every site i."""
    return (x + dx) % L + L * ((y + dy) % L)

ones, mean = np.ones(N), np.full(N, 1 / N)      # couplings; 1/N averages over i

spin, G = [], []
for ry in range(L):
    for rx in range(L):                          # displacement r = (rx, ry)
        i, j = shift(0, 0), shift(rx, ry)
        spin.append(Sz(i, coupling=mean) * Sz(j, coupling=ones)
                    + Splus(i, coupling=mean / 2) * Sminus(j, coupling=ones)
                    + Sminus(i, coupling=mean / 2) * Splus(j, coupling=ones))
        G.append(Cup_dag(i, coupling=mean / 2) * Cup(j, coupling=ones)
                 + Cdn_dag(i, coupling=mean / 2) * Cdn(j, coupling=ones))

key, state, obs = compute_observables(
    key, 2, state, action, wf, N_mc, {"spin": spin, "G": G}, log_every=0,
)

C = obs["spin"].mean(axis=0).reshape(L, L)       # C[ry, rx]
S_k = np.fft.fft2(C).real                        # S_k[m, n] = S(k), k = 2π (n, m) / L
n_k = np.fft.fft2(obs["G"].mean(axis=0).reshape(L, L)).real
```

## References

- A. Nikolaenko, R. Rende, L. L. Viteritti, S. Sachdev and Y.-H. Zhang, "Fermi
  surface change and $d$-wave superconductivity in the square lattice
  Kondo-Heisenberg model", arXiv:2606.23799 (2026),
  [arXiv:2606.23799](https://arxiv.org/abs/2606.23799).
- R. Rende, A. Nikolaenko, L. L. Viteritti, S. Sachdev and Y.-H. Zhang,
  "Transformer neural-network quantum states for lattice models of spins and
  fermions: Application to the ancilla layer model", arXiv:2603.02316 (2026),
  [arXiv:2603.02316](https://arxiv.org/abs/2603.02316).
- A. Chen, Z.-Q. Wan, A. Sengupta, A. Georges and C. Roth, "Neural
  network–augmented Pfaffian wave-functions for scalable simulations of
  interacting fermions", *Proc. Natl. Acad. Sci. U.S.A.* **123**, e2535288123
  (2026), [doi:10.1073/pnas.2535288123](https://doi.org/10.1073/pnas.2535288123).
- A. Chen and C. Roth, "lrux: Fast low-rank updates of determinants and
  Pfaffians in JAX", arXiv:2602.05255 (2026),
  [arXiv:2602.05255](https://arxiv.org/abs/2602.05255).
