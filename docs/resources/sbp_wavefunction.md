# Symmetry-Preserving Backflow Pairing Wave Function

In the doped Hubbard model, antiferromagnetism, charge and spin stripes and
$d$-wave superconductivity compete, with energies so close that the outcome of
a variational calculation can depend on the ansatz. Wave functions that can
break the lattice symmetries, such as backflow determinants, tend to converge
to stripe states, which differ from one initialization to the next and have
weak pairing. The Symmetry-Preserving Backflow Pairing (SBP) wave function
([Rende, Viteritti & Georges, 2026](https://arxiv.org/abs/2608.12465)) is
translation invariant by construction. Optimizations started from different
random parameters reach the same state, below the energies of the stripe
states, and show $d$-wave superconductivity in the $t$-$t'$ Hubbard model at
1/8 doping, on lattices up to $24 \times 24$.

This page builds the SBP wave function in Flax, layer by layer, and optimizes
it with tachys for the $t$-$t'$ Hubbard model,

$$
H = -t \sum_{\langle i,j \rangle, \sigma} \big(c^\dagger_{i\sigma} c_{j\sigma} + \text{h.c.}\big)
    - t' \sum_{\langle\langle i,j \rangle\rangle, \sigma} \big(c^\dagger_{i\sigma} c_{j\sigma} + \text{h.c.}\big)
    + U \sum_i n_{i\uparrow} n_{i\downarrow} ,
$$

on an $L \times L$ square lattice with periodic boundaries, with nearest- and
next-nearest-neighbour hoppings $t = 1$ and $t' = -0.2$, $U = 8$, and
$N_e = (1 - \delta)\,N$ electrons on the $N = L^2$ sites, at doping
$\delta = 1/8$.

## The pair-product wave function

A configuration $n = (n_{1\uparrow}, \dots, n_{N\uparrow}, n_{1\downarrow}, \dots, n_{N\downarrow})$
gives the occupations of the $2N$ modes. The SBP wave function is a pair product,

$$
\Psi(n) = \langle n | \Big( \sum_{i,j=1}^{N} f_{ij}(n)\, c^\dagger_{i\uparrow} c^\dagger_{j\downarrow} \Big)^{N_e/2} | 0 \rangle ,
$$

where $f_{ij}$ is the amplitude for creating a pair of electrons on sites $i$
and $j$, with spins $\uparrow$ and $\downarrow$. Since each pair joins opposite
spins, the amplitude of a configuration with $N_e/2$ electrons of each spin is,
up to a constant factor, a determinant
([Bouchaud, Georges & Lhuillier, 1988](https://doi.org/10.1051/jphys:01988004904055300)):

$$
\Psi(n) = \det \big[ f_{R_\uparrow R_\downarrow}(n) \big] ,
$$

the determinant of the $N_e/2 \times N_e/2$ submatrix of $f(n)$ with the rows
of the occupied $\uparrow$ sites, $R_\uparrow$, and the columns of the occupied
$\downarrow$ sites, $R_\downarrow$.

In a standard pair product, $f$ is a matrix of parameters. In the SBP, a neural
network computes it from the configuration, as a backflow:

$$
f_{ij}(n) = e^{-\lambda\, d(i,j)} \sum_{\alpha=1}^{d} W_{i-j,\alpha}\; y^\uparrow_{i\alpha}(n)\; y^\downarrow_{j\alpha}(n) .
$$

- The backflow vectors $\mathbf{y}^\sigma_i(n) \in \mathbb{R}^d$, one for each
  mode, are the outputs of a transformer ({doc}`vit_wavefunction`). The
  transformer is translation equivariant: translating the configuration
  translates the vectors.
- The weights $W_{i-j,\alpha}$ depend only on the displacement
  $\mathbf{r}_i - \mathbf{r}_j$ between the two sites, for $N \times d$
  parameters.
- $d(i,j)$ is the distance between the two sites, with the minimum image, and
  the decay rate $\lambda$ is trainable. The prefactor suppresses the pairing
  amplitudes of distant sites, which stabilizes the optimization on large
  lattices. It does not limit the range of the pairing correlations: a local
  pairing amplitude can still give off-diagonal long-range order.

## Architecture

The network maps a configuration $n$ to $\log\Psi(n)$ in three steps:

1. **Embedding.** A lookup table shared by all the modes maps each occupation
   $n_{i\sigma} \in \{0, 1\}$ to a vector $\mathbf{x}^\sigma_i \in \mathbb{R}^d$,
   which gives the sequence
   $(\mathbf{x}^\uparrow_1, \dots, \mathbf{x}^\uparrow_N, \mathbf{x}^\downarrow_1, \dots, \mathbf{x}^\downarrow_N)$
   of $2N$ vectors.
2. **Transformer encoder.** A stack of $n_l$ layers with spatial attention and
   real parameters maps the sequence to the backflow vectors
   $(\mathbf{y}^\uparrow_1, \dots, \mathbf{y}^\uparrow_N, \mathbf{y}^\downarrow_1, \dots, \mathbf{y}^\downarrow_N)$,
   followed by a layer normalization.
3. **Pairing layer.** The pairing matrix $f(n)$ from the backflow vectors, and
   its determinant over the occupied sites.

## 1. Spatial attention

The encoder is that of the {doc}`vit_wavefunction`, with one vector for each
mode. The spatial attention combines the $2N$
vectors as

$$
\mathbf{A}^\sigma_i = \sum_{\sigma'} \sum_{j=1}^{N}
\frac{e^{-\gamma_{\sigma\sigma'} d(i,j)}}{\sum_{j'=1}^{N} e^{-\gamma_{\sigma\sigma'} d(i,j')}}\,
\alpha^{\sigma\sigma'}_{i-j}\, V\mathbf{x}^{\sigma'}_j ,
$$

with one kernel for each pair of spin species: trainable weights
$\alpha^{\sigma\sigma'}_{i-j}$, which depend only on the displacement between
the two sites, and a trainable inverse length $\gamma_{\sigma\sigma'}$.

In the code, a displacement is labelled like a site: the displacement
$\mathbf{r}_i - \mathbf{r}_j = (dx, dy)$, modulo $L$, gets the index
`k = dx + L * dy` of the site in column $dx$ and row $dy$
({doc}`../guide/lattices`). `displacements(L)` returns `k[i, j]` for every pair
of sites and the length `d[k]` of every displacement. The weights are stored
per displacement, `alpha[σ, σ', head, k]`; indexing with `k` spreads them over
all pairs of sites, and the `einsum` sums over $\sigma'$ and $j$, as in the
formula.

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
    def __call__(self, x):                        # x: (batch, 2N, d_model), N = L * L
        N = self.L ** 2
        k, d = displacements(self.L)
        alpha = self.param("alpha", nn.initializers.xavier_uniform(),
                           (2, 2, self.n_heads, N), jnp.float64)  # (σ, σ', head, k)
        gamma = self.param("gamma", nn.initializers.ones,
                           (2, 2, self.n_heads, 1), jnp.float64)
        decay = jax.nn.softmax(-gamma * d, axis=-1)     # e^{-γ d_k} / Σ_k' e^{-γ d_k'}
        w = (decay * alpha)[..., k]                     # w[σ, σ', head, i, j]

        V = nn.Dense(self.d_model, kernel_init=nn.initializers.xavier_uniform(),
                     param_dtype=jnp.float64)
        out = nn.Dense(self.d_model, kernel_init=nn.initializers.xavier_uniform(),
                       param_dtype=jnp.float64)
        v = rearrange(V(x), "b (t j) (h e) -> b t h j e", t=2, h=self.n_heads)
        A = jnp.einsum("sthij,bthje->bshie", w, v)      # sum over σ' = t and j
        return out(rearrange(A, "b s h i e -> b (s i) (h e)"))
```

An encoder block applies the attention, then a two-layer feed-forward network,
each after a layer normalization and with a skip connection, as in the ViT:

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

## 2. Pairing layer

The pairing layer builds $f(n)$ from $\mathbf{y}^\uparrow_i$ and
$\mathbf{y}^\downarrow_j$, the outputs of the encoder for the modes
$(i, \uparrow)$ and $(j, \downarrow)$. With `add_dwave_bias=True`, it also adds
a trainable $d$-wave term $\Delta_d\, h_{i-j}$ to $f_{ij}$, with
$h_{i-j} = +1$ for $\mathbf{r}_i - \mathbf{r}_j = \pm\hat{x}$, $-1$ for
$\pm\hat{y}$ and zero otherwise.

The rows and the columns of the determinant are the occupied $\uparrow$ and
$\downarrow$ sites, in increasing order. `_log_det` returns $\log\det A$ as a
complex number, $\log|\det A| + i\arg\det A$: the imaginary part, 0 or $\pi$,
is the sign of the amplitude.

```python
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from tachys.lattice.ansatz.fermionic_transformer import _log_det


def dwave_form_factor(L):
    """h[k]: +1 for the displacements ±x, -1 for ±y, 0 otherwise."""
    h = np.zeros(L * L)
    h[[1, L - 1]] = 1.0                           # (±1, 0)
    h[[L, L * (L - 1)]] = -1.0                    # (0, ±1)
    return h


class PairingHead(nn.Module):
    d_model: int
    L: int
    Ne: int
    add_dwave_bias: bool = False

    @nn.compact
    def __call__(self, y, occupations):          # y: (batch, 2 * L * L, d_model)
        N = self.L ** 2
        k, d = displacements(self.L)

        W = self.param("W", nn.initializers.xavier_uniform(),
                       (self.d_model, N), jnp.float64)
        lam = self.param("lambda", nn.initializers.constant(2.0), (1,), jnp.float64)

        W = (jnp.exp(-lam * d) * W)[:, k]   # e^{-λ d(i,j)} W_{i-j}: (d_model, N, N)
        y_up, y_dn = y[:, :N], y[:, N:]
        f = jnp.einsum("bia,bja,aij->bij", y_up, y_dn, W)
        if self.add_dwave_bias:
            delta_d = self.param("delta_d", nn.initializers.constant(0.5),
                                 (1,), jnp.float64)
            f = f + delta_d * dwave_form_factor(self.L)[k]

        occupied = lambda n: jnp.nonzero(n, size=self.Ne // 2)[0]
        rows = jax.vmap(occupied)(occupations[:, :N])    # (batch, Ne/2): the ↑ sites
        cols = jax.vmap(occupied)(occupations[:, N:])    # (batch, Ne/2): the ↓ sites
        A = jnp.take_along_axis(f, rows[:, :, None], axis=1)
        A = jnp.take_along_axis(A, cols[:, None, :], axis=2)   # (batch, Ne/2, Ne/2)
        return _log_det(A)                               # (batch,)
```

## 3. The SBP wave function

`SBP` chains the three steps. Like every tachys ansatz, it takes a state
holding a batch of configurations, here a `FermionState`, and returns one
log-amplitude per configuration ({doc}`../guide/wavefunctions`). It reads the
lattice size from the configurations and the number of electrons from the
state.

```python
import jax.numpy as jnp
import flax.linen as nn


class SBP(nn.Module):
    num_layers: int
    d_model: int
    n_heads: int
    add_dwave_bias: bool = False

    @nn.compact
    def __call__(self, state):
        n = state.occupations                        # (batch, 2 * L * L)
        L = round((n.shape[-1] // 2) ** 0.5)
        x = nn.Embed(2, self.d_model, embedding_init=nn.initializers.xavier_uniform(),
                     param_dtype=jnp.float64)(n)     # (batch, 2 * L * L, d_model)
        for _ in range(self.num_layers):
            x = EncoderBlock(self.d_model, self.n_heads, L)(x)
        y = nn.LayerNorm(param_dtype=jnp.float64)(x)
        return PairingHead(self.d_model, L, state.Ne, self.add_dwave_bias)(y, n)
```

```python
import jax
from tachys.lattice.lattice_database import square
from tachys.lattice.fermions.fermion_state import FermionState, init_config_spinful

L, Ne = 8, 56
occupations, N_up, N_down = init_config_spinful(jax.random.key(0), Ns=L * L, Ne=Ne,
                                                N_mc=200)
state = FermionState(occupations=occupations, Ne=Ne, lattice=square(shape=(L, L)))

model = SBP(num_layers=2, d_model=32, n_heads=4)
params = model.init(jax.random.key(1), state)
model.apply(params, state).shape        # (200,), complex128
```

## Ground-state optimization

The script follows the {doc}`../quickstart`, on an $8 \times 8$ lattice with
$N_e = 56$ electrons. `hubbard_hamiltonian` takes the hopping amplitude of each
bond direction ({doc}`../guide/hamiltonians`). The SBP has $n_l = 2$ layers,
$h = 4$ heads and $d = 32$, for 25,441 parameters. The pairing matrix is real,
so the sign of $\Psi$ does not depend continuously on the parameters, and the
optimizer runs in real mode ({doc}`../guide/optimization`); it is MARCH, as in
the paper.

```python
import jax
import jax.numpy as jnp

from tachys.lattice.lattice_database import square
from tachys.lattice.fermions.fermion_state import FermionState, init_config_spinful
from tachys.lattice.fermions.hamiltonians.hubbard import hubbard_hamiltonian
from tachys.lattice.bond_exchange import BondExchange
from tachys.wavefunction import WaveFunction
from tachys.optimizer import MARCH
from tachys.montecarlo import sample
from tachys.lattice.operator.local_estimator import compute_expectation

L = 8
lattice = square(shape=(L, L))
N = lattice.Ns
Ne = 56                                 # doping δ = 1 - Ne / N = 1/8

t, tp, U = 1.0, -0.2, 8.0
H = hubbard_hamiltonian(lattice, nn=[
    ((1, 0), t), ((0, 1), t),           # nearest neighbours
    ((1, 1), tp), ((1, -1), tp),        # next-nearest neighbours
], U=U)

key = jax.random.key(0)
N_mc = 512
key, subkey = jax.random.split(key)
occupations, N_up, N_down = init_config_spinful(subkey, Ns=N, Ne=Ne, N_mc=N_mc)
state = FermionState(occupations=occupations, Ne=Ne, lattice=lattice)

model = SBP(num_layers=2, d_model=32, n_heads=4)
key, subkey = jax.random.split(key)
params = model.init(subkey, state)
wf = WaveFunction(params=params, apply_fn=model.apply)
print(wf.num_params)                    # 25441

action = BondExchange.create(lattice, Nbands=2)
optimizer = MARCH(diag_shift=1e-4, mode="real")
opt_state = optimizer.init(wf.params)

N_steps, lr = 100, 0.01
for step in range(N_steps):
    key, subkey = jax.random.split(key)
    mc_keys = jax.random.split(subkey, N_mc)

    state, log_amps, acceptance = sample(1, state, action, mc_keys, wf)
    E_L, e_mean, e2_mean = compute_expectation(H, wf, state, log_amps)

    updates, opt_state = optimizer(E_L, opt_state, state, wf)
    wf = wf.apply_gradients(updates, lr)

    if step % 20 == 0:
        print(f"step {step:3d}  E/N = {jnp.real(e_mean) / N: .5f}")
```

```text
25441
step   0  E/N =  1.72932
step  20  E/N = -0.40595
step  40  E/N = -0.63168
step  60  E/N = -0.66504
step  80  E/N = -0.68803
```

The 100 steps take about 9 minutes on an NVIDIA RTX A6000 GPU, and the
energy is still decreasing. For converged energies, see the Hubbard model on
the
<a href="../benchmarks.html#Hubbard/Square/8%20x%208/t%E2%80%B2%2Ft%20%3D%20%E2%88%920.2/Periodic">benchmarks page</a>.

## Pairing correlations

$d$-wave superconducting order shows in the correlations of pairs at large
distances. Following the paper,

$$
C_p(\mathbf{r}) = \frac{1}{8} \sum_{\boldsymbol{\eta},\boldsymbol{\eta}'} h_{\boldsymbol{\eta}}\, h_{\boldsymbol{\eta}'}
\big\langle c^\dagger_{\boldsymbol{\eta}\downarrow}\, c^\dagger_{\mathbf{0}\uparrow}\, c_{\mathbf{r}\uparrow}\, c_{\mathbf{r}+\boldsymbol{\eta}'\downarrow} \big\rangle
- \mathcal{N}(\mathbf{r}) ,
$$

where $\boldsymbol{\eta}$ and $\boldsymbol{\eta}'$ run over the four
nearest-neighbour vectors, with $h_{\boldsymbol{\eta}} = +1$ for $\pm\hat{x}$
and $-1$ for $\pm\hat{y}$. $\mathcal{N}(\mathbf{r})$ subtracts the disconnected
contribution, a product of the one-body correlations
$G_\sigma(\mathbf{r}) = \langle c^\dagger_{\mathbf{0}\sigma} c_{\mathbf{r}\sigma} \rangle$:

$$
\mathcal{N}(\mathbf{r}) = \frac{1}{8} \sum_{\boldsymbol{\eta},\boldsymbol{\eta}'} h_{\boldsymbol{\eta}}\, h_{\boldsymbol{\eta}'}\,
G_\downarrow(\mathbf{r} + \boldsymbol{\eta}' - \boldsymbol{\eta})\, G_\uparrow(\mathbf{r}) .
$$

A finite value of $C_p$ at large $|\mathbf{r}|$ signals $d$-wave
superconducting order. The mean of $C_p(\mathbf{r})$ over the displacements
longer than half the largest distance of the cluster gives the order
parameter, $\Delta^2_{SC}$.

Each operator below is averaged over the $N$ positions of the origin, with
arrays of sites ({doc}`../guide/hamiltonians`): `shift(dx, dy)` gives the site
$\mathbf{r}_i + (dx, dy)$ for every site $i$. `compute_observables` runs 2
sampling steps, its second argument, and measures all the operators on the
configurations of each step.

```python
import numpy as np
from tachys.lattice.fermions.fermion_operators import Cup, Cup_dag, Cdn, Cdn_dag
from tachys.ground_state_training import compute_observables

x, y = np.arange(N) % L, np.arange(N) // L      # site i = x + L * y

def shift(dx, dy):
    """The site r_i + (dx, dy), for every site i."""
    return (x + dx) % L + L * ((y + dy) % L)

bonds = [((1, 0), 1.0), ((-1, 0), 1.0), ((0, 1), -1.0), ((0, -1), -1.0)]  # η, h_η
ones, mean = np.ones(N), np.full(N, 1 / N)      # couplings; 1/N averages over i

pair, G_up, G_dn = [], [], []
for ry in range(L):
    for rx in range(L):                          # displacement r = (rx, ry)
        op = None
        for (ex, ey), h in bonds:
            for (fx, fy), h2 in bonds:
                term = (Cdn_dag(shift(ex, ey), coupling=h * h2 / 8 * mean)
                        * Cup_dag(shift(0, 0), coupling=ones)
                        * Cup(shift(rx, ry), coupling=ones)
                        * Cdn(shift(rx + fx, ry + fy), coupling=ones))
                op = term if op is None else op + term
        pair.append(op)
        G_up.append(Cup_dag(shift(0, 0), coupling=mean)
                    * Cup(shift(rx, ry), coupling=ones))
        G_dn.append(Cdn_dag(shift(0, 0), coupling=mean)
                    * Cdn(shift(rx, ry), coupling=ones))

key, state, obs = compute_observables(
    key, 2, state, action, wf, N_mc, {"pair": pair, "G_up": G_up, "G_dn": G_dn},
    log_every=0,
)
```

## References

- R. Rende, L. L. Viteritti and A. Georges, "Superconductivity in the $t$-$t'$
  Hubbard model from symmetry-preserving neural-network quantum states",
  arXiv:2608.12465 (2026),
  [arXiv:2608.12465](https://arxiv.org/abs/2608.12465).
- J. Bouchaud, A. Georges and C. Lhuillier, "Pair wave functions for strongly
  correlated fermions and their determinantal representation", *J. Phys.
  France* **49**, 553 (1988),
  [doi:10.1051/jphys:01988004904055300](https://doi.org/10.1051/jphys:01988004904055300).
