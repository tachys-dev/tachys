# Initial Configurations

`sample` advances `N_mc` Markov chains in parallel and needs a starting
configuration for each. The configurations are stored in a `State`, whose
leading axis runs over the chains; {doc}`../concepts` describes the type
itself.

The starting configurations also select the symmetry sector. A move that
conserves a quantum number never changes it — `BondExchange` conserves the
magnetization of spins and the number of electrons of each spin — so the chains
stay in the sector they start in.

## Spins

`SpinState` holds spin-½ configurations as an integer array `spins` of shape
`(N_mc, Ns)`, with $+1$ for $\uparrow$ and $-1$ for $\downarrow$.
`init_config_fixed_magn` draws configurations uniformly at random with total
magnetization $S^z$ equal to `sz`, that is `Ns/2 + sz` spins up and
`Ns/2 - sz` down (`Ns` even):

```python
import jax
from tachys.lattice.lattice_database import square
from tachys.lattice.spins.spin_state import SpinState, init_config_fixed_magn

lattice = square(shape=(4, 4))
key = jax.random.key(0)
N_mc = 1024

spins = init_config_fixed_magn(key, lattice.Ns, sz=0, N_mc=N_mc)  # (N_mc, Ns)
state = SpinState(spins=spins, lattice=lattice)
```

Other sectors are selected with `sz`: for a singlet ground state, for example,
the lowest energy at `sz=1` minus the one at `sz=0` is the spin gap. With a
move that does not conserve $S^z$, such as `SpinFlip`, `sz` only sets where the
chains start.

## Fermions

`FermionState` holds occupation numbers, an integer array `occupations` of
shape `(N_mc, 2 Ns)` with entries 0 and 1, together with the number of
electrons `Ne`. The first `Ns` columns are the $\uparrow$ occupations of sites
$0, \dots, N_s - 1$, the last `Ns` the $\downarrow$ ones:

$$
(n_{0\uparrow}, \dots, n_{N_s-1\,\uparrow},\; n_{0\downarrow}, \dots, n_{N_s-1\,\downarrow}) .
$$

The fermionic operators order their Jordan–Wigner string the same way.

`init_config_spinful` draws random configurations with `Ne/2 + sz` electrons
$\uparrow$ and `Ne/2 - sz` electrons $\downarrow$ (`Ne` even), and returns these
two numbers along with the array:

```python
from tachys.lattice.fermions.fermion_state import (
    FermionState, init_config_spinful,
)

Ne = 14                                 # 1/8 hole doping on 16 sites
occupations, N_up, N_down = init_config_spinful(
    key, Ns=lattice.Ns, Ne=Ne, N_mc=N_mc,
)
state = FermionState(occupations=occupations, Ne=Ne, lattice=lattice)
```

## Custom configurations

Any integer array of the right shape can seed the chains, which covers the
cases the helpers do not, such as an odd number of electrons. To start every
chain from the Néel state:

```python
import jax.numpy as jnp

i, j, _ = lattice.site_coords.T         # row and column of every site
neel = jnp.where((i + j) % 2 == 0, 1, -1).astype(jnp.int8)
state = SpinState(spins=jnp.tile(neel, (N_mc, 1)), lattice=lattice)
```

Chains that start from the same configuration stay correlated until the burn-in
sweeps have separated them, see {doc}`sampling`.

The `state` returned by `sample` is itself a valid starting point: the training
loop feeds it back at every step, and checkpoints store it.

## Number of chains

With several devices, the chains are split evenly among them, so `N_mc` must be
a multiple of the number of devices. The same `N_mc` sets the size of the
sample for every energy estimate, and of the linear system solved by the
optimizer ({doc}`optimization`).
