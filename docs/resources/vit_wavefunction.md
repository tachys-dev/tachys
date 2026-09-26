# Vision Transformer Wave Function

Transformers ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)) are the
leading architecture in natural language processing and, as Vision Transformers
(ViT), in computer vision
([Dosovitskiy et al., 2021](https://arxiv.org/abs/2010.11929)). Their core is
the attention mechanism, which builds the representation of each element of a
sequence from all the others, and can therefore describe correlations at any
distance.

This page builds the ViT wave function for spin-½ lattice models
([Viteritti, Rende & Becca, 2023](https://doi.org/10.1103/PhysRevLett.130.236401);
[Viteritti et al., 2025](https://doi.org/10.1103/PhysRevB.111.134411)) layer by
layer in Flax, with the spatial attention mechanism
([Viteritti, Rende, Sachdev & Carleo, 2026](https://arxiv.org/abs/2602.02665)),
and optimizes it with tachys for the $J_1$-$J_2$ Heisenberg model. With factored
attention in place of spatial attention, the network is `SpinViT`
(`tachys.lattice.ansatz.spin_vit`).

## Architecture

The input is a configuration $\sigma = (\sigma_1, \dots, \sigma_N)$,
$\sigma_r = \pm 1$, of the $N = L^2$ spins of an $L \times L$ lattice. The ViT
maps it to $\log\psi(\sigma)$ in three steps:

1. **Embedding.** The lattice is cut into $n = N/b^2$ patches of $b \times b$
   sites, and one linear map, shared by all patches, sends the spins of patch
   $i$ to a vector $\mathbf{x}_i \in \mathbb{R}^d$.
2. **Transformer encoder.** A stack of layers with real parameters maps the
   sequence $(\mathbf{x}_1, \dots, \mathbf{x}_n)$ to
   $(\mathbf{y}_1, \dots, \mathbf{y}_n)$, with $\mathbf{y}_i \in \mathbb{R}^d$.
   While $\mathbf{x}_i$ depends only on the spins of patch $i$, $\mathbf{y}_i$
   depends on all the spins.
3. **Output layer.** A fully connected layer with complex parameters maps the
   hidden representation $\mathbf{z} = \sum_{i=1}^{n} \mathbf{y}_i$ to
   $\log\psi(\sigma)$.

## 1. Embedding

A batch of configurations has shape `(batch, L * L)`: the spin of the site in
column $x$ and row $y$ is at index $x + L\,y$ ({doc}`../guide/lattices`).
`extract_patches2d` cuts each configuration into $b \times b$ patches: patch
$i = (L/b)\,P + Q$ holds the sites with $\lfloor y/b \rfloor = P$ and
$\lfloor x/b \rfloor = Q$. `Embed` applies the same dense layer to every patch.
As in the built-in ansätze, all parameters are stored in double precision.

```python
import jax.numpy as jnp
import flax.linen as nn


def extract_patches2d(spins, b):
    """(batch, L*L) -> (batch, n_patches, b*b): b × b patches of the lattice."""
    batch, N = spins.shape
    side = round((N // b**2) ** 0.5)             # patches per side, L // b
    x = spins.reshape(batch, side, b, side, b)   # [y, x] -> [P, y % b, Q, x % b]
    x = x.transpose(0, 1, 3, 2, 4)               # [P, Q, y % b, x % b]
    return x.reshape(batch, side * side, b * b)  # patch i = side * P + Q


class Embed(nn.Module):
    d_model: int
    b: int

    @nn.compact
    def __call__(self, spins):
        x = extract_patches2d(spins, self.b)
        return nn.Dense(self.d_model, kernel_init=nn.initializers.xavier_uniform(),
                        param_dtype=jnp.float64)(x)
```

With $10 \times 10$ spins and $2 \times 2$ patches, a configuration becomes a
sequence of 25 vectors, here in $\mathbb{R}^{32}$:

```python
import jax
from tachys.lattice.spins.spin_state import init_config_fixed_magn

spins = init_config_fixed_magn(jax.random.key(0), 100, N_mc=200)  # 200 × 100 spins
embed = Embed(d_model=32, b=2)
params = embed.init(jax.random.key(1), spins)
embed.apply(params, spins).shape        # (200, 25, 32)
```

## 2. Transformer encoder

The encoder combines four ingredients: attention, a two-layer feed-forward
network, layer normalization and skip connections.

### Spatial attention

The attention maps the sequence $(\mathbf{x}_1, \dots, \mathbf{x}_n)$ to a
sequence $(\mathbf{A}_1, \dots, \mathbf{A}_n)$, each
$\mathbf{A}_i \in \mathbb{R}^d$ combining all the inputs. Spatial attention
reads

$$
\mathbf{A}_i = \sum_{j=1}^{n}
\frac{e^{-\gamma d_{ij}}}{\sum_{j'=1}^{n} e^{-\gamma d_{ij'}}}\,
\alpha_{ij}\, V\mathbf{x}_j ,
$$

with trainable attention weights $\alpha_{ij}$, value map $V$ and inverse
length $\gamma$, and $d_{ij}$ the distance between patches $i$ and $j$.

- **Attention weights.** In standard attention, $\alpha_{ij}$ is computed from
  $\mathbf{x}_i$ and $\mathbf{x}_j$ through queries and keys. Here, as in
  factored attention, the weights are parameters: they depend on the positions
  of the two patches, not on their spins. For the ground states of spin
  Hamiltonians, this is as accurate as standard attention, and cheaper
  ([Rende & Viteritti, 2025](https://doi.org/10.1088/2632-2153/ada1a0)). For a
  translation-invariant Hamiltonian, $\alpha_{ij}$ is taken to depend only on
  the displacement from patch $i$ to patch $j$.
- **Distance kernel.** $d_{ij}$ is the Euclidean distance between the two
  patches on the periodic lattice (minimum image), in units of the patch side
  $b$. Like $\alpha_{ij}$, it depends only on the displacement, so the
  attention commutes with the translations of the patch grid. The kernel
  suppresses the contributions of patches farther than about $1/\gamma$. It is
  a soft prior: $\gamma$ is learned, from $\gamma = 3$ at initialization, every
  patch still attends to every other, and for $\gamma \to 0$ the kernel tends to
  $1/n$, which gives back factored attention.

Optimized transformer wave functions typically learn attention weights that
decay with the distance, but learning this decay from a random initialization
becomes harder as the lattice grows. Built into the architecture, it stabilizes the
optimization on large lattices: up to $42 \times 42$ sites in
[Viteritti, Rende, Sachdev & Carleo (2026)](https://arxiv.org/abs/2602.02665).

*Multi-head* attention computes $h$ such maps in parallel, each with its own
$\alpha_{ij}$, $\gamma$ and $V$, with values in $\mathbb{R}^{d/h}$, so that
different heads can capture different length scales. The $h$ outputs are
concatenated into a vector of $\mathbb{R}^d$ and mixed by a linear map $W$.

In the code, `side` $= L/b$ is the number of patches per side, and each head
stores one weight per displacement, `alpha[mu, k]`, and one inverse length,
`gamma[mu]`. `displacement_index(side)[i, j]` is the displacement `k` from patch
`i` to patch `j`, and `displacement_length(side)[k]` its length. Every patch
sees each displacement once, so the denominator is the same for all $i$.
Indexing with `displacement_index` then turns the per-displacement rows into
the $n \times n$ matrices of the equation above.

```python
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from einops import rearrange


def displacement_index(side):
    """k[i, j]: displacement from patch i to patch j on the periodic side × side
    grid of patches, k = side * (row shift) + (column shift)."""
    row, col = np.divmod(np.arange(side**2), side)
    d_row = (row[None, :] - row[:, None]) % side
    d_col = (col[None, :] - col[:, None]) % side
    return side * d_row + d_col                  # (n_patches, n_patches)


def displacement_length(side):
    """d[k]: length of displacement k, minimum image, in units of the patch side."""
    d_row, d_col = np.divmod(np.arange(side**2), side)
    return np.hypot(np.minimum(d_row, side - d_row), np.minimum(d_col, side - d_col))


class SpatialAttention(nn.Module):
    d_model: int
    n_heads: int
    n_patches: int
    gamma_init: float = 3.0

    @nn.compact
    def __call__(self, x):                       # x: (batch, n_patches, d_model)
        side = round(self.n_patches ** 0.5)
        k, d = displacement_index(side), displacement_length(side)

        # per head: one weight per displacement, and one inverse length
        alpha = self.param("alpha", nn.initializers.xavier_uniform(),
                           (self.n_heads, self.n_patches), jnp.float64)
        gamma = self.param("gamma", nn.initializers.constant(self.gamma_init),
                           (self.n_heads, 1), jnp.float64)
        decay = jax.nn.softmax(-gamma * d, axis=-1)  # e^{-γ d_k} / Σ_k' e^{-γ d_k'}
        weights = (decay * alpha)[:, k]              # (n_heads, n_patches, n_patches)

        V = nn.Dense(self.d_model, kernel_init=nn.initializers.xavier_uniform(),
                     param_dtype=jnp.float64, name="V")
        W = nn.Dense(self.d_model, kernel_init=nn.initializers.xavier_uniform(),
                     param_dtype=jnp.float64, name="W")
        v = rearrange(V(x), "batch j (heads e) -> batch heads j e",
                      heads=self.n_heads)
        A = weights @ v                              # A_i = Σ_j weights_ij V x_j
        return W(rearrange(A, "batch heads i e -> batch i (heads e)"))
```

The output has the shape of the input:

```python
import jax

x = jax.random.normal(jax.random.key(0), (200, 25, 32))  # 200 sequences of 25 vectors
attention = SpatialAttention(d_model=32, n_heads=8, n_patches=25)
params = attention.init(jax.random.key(1), x)
attention.apply(params, x).shape        # (200, 25, 32)
```

At initialization, the kernel of a patch gives 78% of the weight to the patch
itself and 4% to each of its four nearest neighbours:

```python
import jax
import jax.numpy as jnp

decay = jax.nn.softmax(-3.0 * displacement_length(5))       # γ = 3, 5 × 5 patches
print(jnp.roll(decay.reshape(5, 5), (2, 2), axis=(0, 1)).round(3))  # patch at the centre
```

```text
[[0.    0.001 0.002 0.001 0.   ]
 [0.001 0.011 0.039 0.011 0.001]
 [0.002 0.039 0.783 0.039 0.002]
 [0.001 0.011 0.039 0.011 0.001]
 [0.    0.001 0.002 0.001 0.   ]]
```

### Encoder block

An encoder block applies the attention, then a two-layer feed-forward network
to each patch separately, with hidden dimension $4d$ and a GELU activation.
Both are preceded by layer normalization and wrapped in a skip connection,

$$
\mathbf{x} \leftarrow \mathbf{x} + \mathrm{Attention}\big(\mathrm{LN}(\mathbf{x})\big),
\qquad
\mathbf{x} \leftarrow \mathbf{x} + \mathrm{FFN}\big(\mathrm{LN}(\mathbf{x})\big),
$$

which keeps deep stacks trainable. The encoder chains `num_layers` blocks.

```python
import jax.numpy as jnp
import flax.linen as nn


class EncoderBlock(nn.Module):
    d_model: int
    n_heads: int
    n_patches: int

    @nn.compact
    def __call__(self, x):
        y = nn.LayerNorm(param_dtype=jnp.float64)(x)
        x = x + SpatialAttention(self.d_model, self.n_heads, self.n_patches)(y)

        y = nn.LayerNorm(param_dtype=jnp.float64)(x)
        y = nn.Dense(4 * self.d_model, kernel_init=nn.initializers.xavier_uniform(),
                     param_dtype=jnp.float64)(y)
        y = nn.Dense(self.d_model, kernel_init=nn.initializers.xavier_uniform(),
                     param_dtype=jnp.float64)(nn.gelu(y))
        return x + y


class Encoder(nn.Module):
    num_layers: int
    d_model: int
    n_heads: int
    n_patches: int

    @nn.compact
    def __call__(self, x):
        for _ in range(self.num_layers):
            x = EncoderBlock(self.d_model, self.n_heads, self.n_patches)(x)
        return x
```

```python
import jax

x = jax.random.normal(jax.random.key(0), (200, 25, 32))
encoder = Encoder(num_layers=4, d_model=32, n_heads=8, n_patches=25)
params = encoder.init(jax.random.key(1), x)
encoder.apply(params, x).shape          # (200, 25, 32)
```

## 3. Output layer

The output layer maps the hidden representation
$\mathbf{z} = \sum_{i=1}^{n} \mathbf{y}_i$ to the log-amplitude,

$$
\log\psi(\sigma) = \sum_{\alpha=1}^{d} \log\cosh\big(b_\alpha + \mathbf{w}_\alpha \cdot \mathbf{z}\big),
$$

with complex parameters $b_\alpha$ and $\mathbf{w}_\alpha$, while those of the
encoder are real: the encoder maps the configurations to a feature space in
which a single complex layer predicts both modulus and phase
([Viteritti et al., 2025](https://doi.org/10.1103/PhysRevB.111.134411)). The
tachys optimizers take real parameters ({doc}`../guide/optimization`), so the
code computes the real and imaginary parts of
$b_\alpha + \mathbf{w}_\alpha \cdot \mathbf{z}$ with two real dense layers. As
in `SpinViT`, $\mathbf{z}$ and both parts go through layer normalization.

`ViT` chains the three steps. Like every tachys ansatz, it takes a `SpinState`,
holding a batch of configurations, and returns one log-amplitude per
configuration ({doc}`../guide/wavefunctions`).

```python
import jax.numpy as jnp
import flax.linen as nn
from tachys.lattice.ansatz.rbm import log_cosh


class OutputHead(nn.Module):
    d_model: int

    @nn.compact
    def __call__(self, y):                       # y: (batch, n_patches, d_model)
        z = nn.LayerNorm(param_dtype=jnp.float64)(y.sum(axis=1))
        re = nn.Dense(self.d_model, kernel_init=nn.initializers.xavier_uniform(),
                      param_dtype=jnp.float64)(z)
        im = nn.Dense(self.d_model, kernel_init=nn.initializers.xavier_uniform(),
                      param_dtype=jnp.float64)(z)
        re = nn.LayerNorm(param_dtype=jnp.float64)(re)
        im = nn.LayerNorm(param_dtype=jnp.float64)(im)
        return jnp.sum(log_cosh(re + 1j * im), axis=-1)


class ViT(nn.Module):
    num_layers: int
    d_model: int
    n_heads: int
    b: int

    @nn.compact
    def __call__(self, state):
        spins = state.spins                      # (batch, L*L)
        n_patches = spins.shape[-1] // self.b**2
        x = Embed(self.d_model, self.b)(spins)
        y = Encoder(self.num_layers, self.d_model, self.n_heads, n_patches)(x)
        return OutputHead(self.d_model)(y)       # (batch,)
```

```python
import jax
from tachys.lattice.lattice_database import square
from tachys.lattice.spins.spin_state import SpinState, init_config_fixed_magn

L = 10
spins = init_config_fixed_magn(jax.random.key(0), L * L, N_mc=200)
state = SpinState(spins=spins, lattice=square(shape=(L, L)))

model = ViT(num_layers=4, d_model=32, n_heads=8, b=2)
params = model.init(jax.random.key(1), state)
model.apply(params, state).shape        # (200,), complex128
```

## Ground-state optimization

The $J_1$-$J_2$ Heisenberg model on a $10 \times 10$ square lattice with
periodic boundaries,

$$
H = J_1 \sum_{\langle i,j \rangle} \mathbf{S}_i \cdot \mathbf{S}_j
  + J_2 \sum_{\langle\langle i,j \rangle\rangle} \mathbf{S}_i \cdot \mathbf{S}_j ,
$$

at the highly frustrated point $J_2/J_1 = 0.5$, is a standard benchmark for
variational wave functions (see the
<a href="../benchmarks.html">benchmarks page</a>). The script follows the
{doc}`../quickstart`. The ViT has 4 layers with $h = 10$ heads, $d = 60$ and
$2 \times 2$ patches, for 155,660 parameters. No sign rule is imposed: the
network learns the phase, so SR runs in complex mode
({doc}`../guide/optimization`).

```python
import jax
import jax.numpy as jnp

from tachys.lattice.lattice_database import square
from tachys.lattice.spins.spin_state import SpinState, init_config_fixed_magn
from tachys.lattice.spins.hamiltonians.heisenberg import heisenberg_hamiltonian
from tachys.lattice.bond_exchange import BondExchange
from tachys.wavefunction import WaveFunction
from tachys.optimizer import SR
from tachys.montecarlo import sample
from tachys.lattice.operator.local_estimator import compute_expectation

L = 10
lattice = square(shape=(L, L))
N = lattice.Ns

J1, J2 = 1.0, 0.5
H = heisenberg_hamiltonian(lattice, nn=[
    ((1, 0), J1), ((0, 1), J1),         # nearest neighbours
    ((1, 1), J2), ((1, -1), J2),        # next-nearest neighbours
])

key = jax.random.key(0)
N_mc = 1024
key, subkey = jax.random.split(key)
spins = init_config_fixed_magn(subkey, N, N_mc=N_mc)
state = SpinState(spins=spins, lattice=lattice)

model = ViT(num_layers=4, d_model=60, n_heads=10, b=2)
key, subkey = jax.random.split(key)
params = model.init(subkey, state)
wf = WaveFunction(params=params, apply_fn=model.apply)
print(wf.num_params)                    # 155660

action = BondExchange.create(lattice, max_dist=2)
optimizer = SR(diag_shift=1e-4, mode="complex")
opt_state = optimizer.init(wf.params)

N_steps, lr = 100, 0.0075
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
155660
step   0  E/N =  0.50466
step  20  E/N = -0.32359
step  40  E/N = -0.38516
step  60  E/N = -0.40919
step  80  E/N = -0.42281
```

The energy of the optimized state, averaged over 10 batches of `N_mc`
configurations separated by five sweeps, with the standard error of the batch
means:

```python
import numpy as np

energies = []
for _ in range(10):
    key, subkey = jax.random.split(key)
    mc_keys = jax.random.split(subkey, N_mc)
    state, log_amps, acceptance = sample(5, state, action, mc_keys, wf)
    E_L, e_mean, e2_mean = compute_expectation(H, wf, state, log_amps)
    energies.append(float(jnp.real(e_mean)) / N)
error = np.std(energies, ddof=1) / np.sqrt(len(energies))
print(f"E/N = {np.mean(energies):.5f} ± {error:.5f}")
```

```text
E/N = -0.42903 ± 0.00023
```

The 100 steps take about 25 minutes on the CPU of a laptop (Apple M4 Max), and
the energy is still decreasing. A larger ViT, optimized for longer with more
samples, reaches $E/N = -0.497634$ on this lattice
([Rende et al., 2024](https://doi.org/10.1038/s42005-024-01732-4)); the
<a href="../benchmarks.html">benchmarks page</a> lists the energies of other
variational methods.

## Restoring the symmetries

The ViT is invariant under the translations of the patch grid, by multiples of
$b$ sites, but not under a translation by one site. The amplitudes are compared
through their ratio, since a log-amplitude is defined up to a multiple of
$2\pi i$:

```python
import jax.numpy as jnp

log_psi = wf.apply_fn(wf.params, state)
spins = state.spins.reshape(-1, L, L)                          # spins[:, y, x]
by_two = state.replace(spins=jnp.roll(spins, 2, axis=2).reshape(-1, L * L))
by_one = state.replace(spins=jnp.roll(spins, 1, axis=2).reshape(-1, L * L))
ratio_two = jnp.exp(wf.apply_fn(wf.params, by_two) - log_psi)  # ψ(Tσ) / ψ(σ)
ratio_one = jnp.exp(wf.apply_fn(wf.params, by_one) - log_psi)
print(jnp.allclose(ratio_two, 1))                              # True
print(jnp.allclose(ratio_one, 1))                              # False
```

The ground state of the model on this lattice has zero momentum and is
invariant under the point group $C_{4v}$ of rotations and reflections. Summing
the amplitudes over the $b^2$ translations within a patch and over $C_{4v}$,

$$
\psi_{\mathrm{sym}}(\sigma) = \sum_{g \in C_{4v}} \; \sum_{t_x,\, t_y = 0}^{b - 1}
\psi\big(g\, T_{t_x, t_y}\, \sigma\big),
$$

where $T_{t_x, t_y}$ translates the configuration by $t_x$ columns and $t_y$
rows, projects the ViT onto this sector. Together with the invariance under
translations by $b$ sites, the sum makes $\psi_{\mathrm{sym}}$ invariant under
every translation, rotation and reflection of the lattice. `symmetrize_wf`
evaluates the 32 terms ({doc}`../guide/wavefunctions`), and the translation by
one site now leaves the amplitudes unchanged:

```python
import numpy as np
import jax.numpy as jnp
from tachys.lattice.lattice_symmetries import translation_group, point_group
from tachys.lattice.symmetries import combine_perm_groups, symmetrize_wf

translations, shifts = translation_group(lattice)               # the L² translations
translations = translations[np.all(shifts < 2, axis=1)]         # the 4 within a patch
point_ops = np.stack([op.perm for op in point_group(lattice)])  # C4v: 8 operations
perms, _ = combine_perm_groups(translations, point_ops)         # 32 permutations
wf_sym = WaveFunction(params=wf.params, apply_fn=symmetrize_wf(model.apply, perms))

spins = state.spins.reshape(-1, L, L)                           # spins[:, y, x]
by_one = state.replace(spins=jnp.roll(spins, 1, axis=2).reshape(-1, L * L))
ratio = jnp.exp(wf_sym.apply_fn(wf_sym.params, by_one)
                - wf_sym.apply_fn(wf_sym.params, state))        # ψ_sym(Tσ) / ψ_sym(σ)
print(jnp.allclose(ratio, 1))                                   # True
```

`wf_sym` shares its parameters with `wf`, and the loop above optimizes it with
`wf_sym` in place of `wf`, at the cost of 32 evaluations of the network per
amplitude. [Viteritti, Rende, Sachdev & Carleo (2026)](https://arxiv.org/abs/2602.02665)
optimize in three stages: without symmetries, then with the translations, then
with the full symmetry group.

## References

- A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez,
  Ł. Kaiser and I. Polosukhin, "Attention is all you need", *Advances in
  Neural Information Processing Systems* **30** (2017),
  [arXiv:1706.03762](https://arxiv.org/abs/1706.03762).
- A. Dosovitskiy et al., "An image is worth 16x16 words: Transformers for image
  recognition at scale", *International Conference on Learning
  Representations* (2021), [arXiv:2010.11929](https://arxiv.org/abs/2010.11929).
- L. L. Viteritti, R. Rende and F. Becca, "Transformer variational wave
  functions for frustrated quantum spin systems", *Phys. Rev. Lett.* **130**,
  236401 (2023),
  [doi:10.1103/PhysRevLett.130.236401](https://doi.org/10.1103/PhysRevLett.130.236401).
- R. Rende, L. L. Viteritti, L. Bardone, F. Becca and S. Goldt, "A simple
  linear algebra identity to optimize large-scale neural network quantum
  states", *Commun. Phys.* **7**, 260 (2024),
  [doi:10.1038/s42005-024-01732-4](https://doi.org/10.1038/s42005-024-01732-4).
- L. L. Viteritti, R. Rende, A. Parola, S. Goldt and F. Becca, "Transformer
  wave function for two dimensional frustrated magnets: Emergence of a
  spin-liquid phase in the Shastry-Sutherland model", *Phys. Rev. B* **111**,
  134411 (2025),
  [doi:10.1103/PhysRevB.111.134411](https://doi.org/10.1103/PhysRevB.111.134411).
- R. Rende and L. L. Viteritti, "Are queries and keys always relevant? A case
  study on transformer wave functions", *Mach. Learn.: Sci. Technol.* **6**,
  010501 (2025),
  [doi:10.1088/2632-2153/ada1a0](https://doi.org/10.1088/2632-2153/ada1a0).
- L. L. Viteritti, R. Rende, S. Sachdev and G. Carleo, "Approaching the
  thermodynamic limit with neural-network quantum states", arXiv:2602.02665
  (2026), [arXiv:2602.02665](https://arxiv.org/abs/2602.02665).
