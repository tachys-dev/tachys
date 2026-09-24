# Wavefunctions

The variational wavefunction is a Flax module that maps a batch of
configurations to the logarithm of their amplitudes, $\log\psi(x)$: one complex
number per configuration, whose real part is $\log|\psi(x)|$ and whose
imaginary part is the phase. tachys never calls the module directly.
`WaveFunction` pairs the parameters with the module's `apply` function, and
sampling, local energies and optimizers only evaluate
`wf.apply_fn(wf.params, state)`:

```python
from tachys.wavefunction import WaveFunction

params = model.init(key, state)          # shapes set by the configurations
wf = WaveFunction(params=params, apply_fn=model.apply)
log_psi = wf.apply_fn(wf.params, state)  # (N_mc,)
```

## Built-in ansätze

| Module | Configurations | Architecture |
|---|---|---|
| `SpinRBM(hidden_units, complex=False)` | spins | restricted Boltzmann machine |
| `SpinViT(num_layers, d_model, num_heads, seq_len, b, ...)` | spins | vision transformer |
| `FermionRBM(hidden_units)` | fermions | Slater determinant with a one-layer backflow |
| `FermionicTransformer(num_layers, d_model, num_heads, Ne, Ns, ...)` | fermions | Slater determinant with a transformer backflow |
| `SpinFoundationRBM`, `FermionFoundationRBM` | foundation states | RBMs conditioned on the couplings, see {doc}`../foundation_models` |

They live in `tachys.lattice.ansatz`, in the modules `rbm`, `spin_vit`,
`fermionic_transformer` and `rbm_foundation`; the {doc}`../api` lists their
parameters.

`SpinViT` splits a configuration into patches — `b` consecutive sites in one
dimension, or `b × b` squares on an $L \times L$ lattice with
`two_dimensional=True` — and `seq_len` is the number of patches, `Ns // b` or
`Ns // b**2`. With `transl_invariant=True`, the attention between two patches
depends only on their separation. `complex=True`, the default, adds a second
output head for the phase.

## Writing an ansatz

Any `flax.linen.Module` works, provided `__call__` takes a batch of
configurations, a `State` with a leading batch axis, and returns one
log-amplitude per configuration. The module never sees a single configuration:
when tachys evaluates one, as the optimizers do for the per-sample Jacobians,
`WaveFunction` passes it on as a batch of one. `model.init` is called directly,
so give it a batch too. A Jastrow wavefunction,
$\log\psi(\sigma) = \sum_{i,j} \sigma_i W_{ij}\, \sigma_j$, takes a few lines:

```python
import jax.numpy as jnp
import flax.linen as nn

class Jastrow(nn.Module):
    @nn.compact
    def __call__(self, state):
        s = state.spins.astype(jnp.float64)          # (batch, Ns)
        Ns = s.shape[-1]
        init = nn.initializers.normal(0.01)
        W = self.param("W", init, (Ns, Ns), jnp.float64)
        return jnp.einsum("bi,ij,bj->b", s, W, s)    # (batch,)

model = Jastrow()
params = model.init(key, state)
wf = WaveFunction(params=params, apply_fn=model.apply)
```

## Signs and phases

The Jastrow wavefunction is positive. That suffices when the ground state has
no sign structure in the $S^z$ basis, or when the signs are known in advance
and can be supplied by hand. On bipartite lattices, the Marshall sign rule
gives the signs of the ground state of the nearest-neighbour Heisenberg model,
and `add_sign_rule` adds it to the log-amplitude:

```python
from tachys.lattice.spins.sign_rules import add_sign_rule, MSR_log_phase_square

apply_fn = add_sign_rule(MSR_log_phase_square, model.apply, L)
wf = WaveFunction(params=params, apply_fn=apply_fn)
```

`MSR_log_phase_square` takes the linear size `L` of an $L \times L$ cluster;
`tachys.lattice.spins.sign_rules` also provides the Marshall rule of the chain
and the 120° rule of the triangular lattice. Otherwise the phase is learned,
which takes a complex output such as those of `SpinViT` and
`SpinRBM(complex=True)`; for fermions, the Slater determinant carries the sign.
Whether the phase depends on the parameters decides the `mode` of the
optimizer, see {doc}`optimization`.

## Symmetries

`tachys.lattice.symmetries` wraps an apply function so that the wavefunction
belongs to a symmetry sector:

- `symmetrize_wf(apply_fn, perms)` sums the amplitudes over a group of site
  permutations, such as the translations returned by
  `translation_group(lattice)` or the rotations and reflections returned by
  `point_group(lattice)`, both in `tachys.lattice.lattice_symmetries`;
- `singlet_symm(apply_fn)` symmetrizes a spin wavefunction under the global
  spin flip;
- `spin_flip_symm_f(apply_fn, p)` does the same for spinful fermions, in the
  sector `p = ±1`;
- `time_reversal(apply_fn)` makes the wavefunction real.

```python
import numpy as np
from tachys.lattice.lattice_symmetries import point_group
from tachys.lattice.symmetries import symmetrize_wf

perms = np.stack([op.perm for op in point_group(lattice)])  # C4v: 8 operations
wf = WaveFunction(params=params, apply_fn=symmetrize_wf(model.apply, perms))
```

The symmetrized wavefunction evaluates the network once per group element. For
fermions, `expand_perm(perms, 2)` extends the site permutations to both spin
species, and `symmetrize_wf` adds the fermionic signs.
