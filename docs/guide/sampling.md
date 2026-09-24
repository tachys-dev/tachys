# Monte Carlo Sampling

`sample` draws configurations from $|\psi(x)|^2$ with the Metropolis–Hastings
algorithm. At each step, a move proposes a new configuration $x'$ on every
chain, and the chain accepts it with probability

$$
A(x \to x') = \min\left(1,\; \frac{|\psi(x')|^2}{|\psi(x)|^2}\,\frac{q(x \mid x')}{q(x' \mid x)}\right),
$$

where $q(x' \mid x)$ is the probability that the move proposes $x'$ from $x$.

```python
from tachys.montecarlo import sample

key, subkey = jax.random.split(key)
mc_keys = jax.random.split(subkey, N_mc)          # one key per chain
state, log_amps, acceptance = sample(nsweeps, state, action, mc_keys, wf)
```

A sweep is `Ns` steps, so each chain receives `nsweeps × Ns` proposals. `sample`
returns the new configurations, their log-amplitudes, which
`compute_expectation` reuses, and the fraction of accepted proposals.

## Choosing a move

A move must let the chains reach every configuration of the sector, and keep
the quantum numbers the Hamiltonian conserves — otherwise the chains leave the
sector set by the {doc}`initial configurations <configurations>`.

| Move | Configurations | Proposal | Conserves |
|---|---|---|---|
| `BondExchange.create(lattice)` | spins | swap two antiparallel spins on a bond | $S^z$ |
| `BondExchange.create(lattice, Nbands=2)` | fermions | move an electron along a bond, keeping its spin | $N_\uparrow$, $N_\downarrow$ |
| `FermionSpinExchange.create(lattice)` | fermions | swap the spins of two singly occupied sites on a bond | $N_\uparrow$, $N_\downarrow$ |
| `SpinFlip()` | spins | flip one spin | — |
| `BondFlip.create(lattice, deltas)` | spins | flip both spins on a bond | parity of $N_\uparrow$ |

`BondExchange` lives in `tachys.lattice.bond_exchange`, `SpinFlip` and
`BondFlip` in `tachys.lattice.spins.spin_action`, `FermionSpinExchange` in
`tachys.lattice.fermions.fermion_action`.

`BondExchange` and `FermionSpinExchange` draw their bonds from the distance
shells of the lattice ({doc}`lattices`): `max_dist=1`, the default, uses
nearest neighbours, and `max_dist=2` adds the next shell, which helps when the
Hamiltonian couples further neighbours. `BondFlip` takes cell displacements, as
`lattice.bonds` does.

For fermions at strong coupling and close to half filling, an electron that
moves usually creates a doubly occupied site and is rejected, so the spins
rearrange slowly.
`FermionSpinExchange` exchanges spins without moving charge. `CompositeAction`
mixes moves, choosing one on each chain at each step with the given
probabilities:

```python
from tachys.montecarlo import CompositeAction
from tachys.lattice.bond_exchange import BondExchange
from tachys.lattice.fermions.fermion_action import FermionSpinExchange

action = CompositeAction(
    actions=(
        BondExchange.create(lattice, Nbands=2),
        FermionSpinExchange.create(lattice),
    ),
    probs=(0.5, 0.5),
)
```

With a composite move, `acceptance` has one entry per move.

## Sweeps, burn-in and acceptance

Successive configurations of a chain are correlated, and `nsweeps` sets how
many proposals separate two samples. The chains carry over from one call to the
next, so a single sweep per optimization step is often enough; the quickstart
uses `nsweeps=1`.

The initial configurations are not distributed according to $|\psi|^2$. Before
measuring anything, and at the start of training, a burn-in call brings the
chains to equilibrium:

```python
key, subkey = jax.random.split(key)
mc_keys = jax.random.split(subkey, N_mc)
state, log_amps, acceptance = sample(20, state, action, mc_keys, wf)
```

A low acceptance means the chains move slowly: increase `nsweeps`, or choose a
move better suited to the wavefunction.

## Writing a move

A move is a subclass of `_BaseAction` (`tachys.montecarlo`). Its
`__call__(key, state)` receives one PRNG key per chain and the batch of
configurations, and returns three values:

- `new_state`, the proposed configurations;
- `allowed_move`, one boolean per chain: `False` rejects the proposal without
  evaluating the wavefunction, for proposals that leave the configuration
  unchanged or violate a constraint;
- `log_prob_correction`, the logarithm of $q(x \mid x')/q(x' \mid x)$, or `0.0`
  for a symmetric proposal.

This move swaps two sites drawn anywhere on the lattice. It conserves $S^z$,
like `BondExchange`, but also connects distant sites in a single step:

```python
import jax
import jax.numpy as jnp
from tachys.montecarlo import _BaseAction

class RandomExchange(_BaseAction):
    def __call__(self, key, state):
        spins = state.spins                          # (N_mc, Ns)
        chains = jnp.arange(spins.shape[0])
        Ns = spins.shape[1]
        pick = lambda k: jax.random.choice(k, Ns, (2,), replace=False)
        i, j = jax.vmap(pick)(key).T                 # two distinct sites
        si, sj = spins[chains, i], spins[chains, j]
        new_spins = spins.at[chains, i].set(sj).at[chains, j].set(si)
        allowed = si != sj           # swapping equal spins changes nothing
        return state.replace(spins=new_spins), allowed, 0.0

action = RandomExchange()
```

The pair is drawn uniformly, so the proposal is symmetric and the correction
vanishes.
