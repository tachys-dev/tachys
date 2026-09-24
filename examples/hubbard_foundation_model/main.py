
import jax
import jax.numpy as jnp
from tachys.lattice.ansatz.rbm_foundation import FermionFoundationRBM
from tachys.lattice.bond_exchange import BondExchange
from tachys.lattice.fermions.fermion_operators import Cdn, Cdn_dag, Cup, Cup_dag, Ndn, Nup
from tachys.lattice.fermions.fermion_state import init_config_spinful
from tachys.lattice.foundation.foundation_state import FermionFoundationState
from tachys.lattice.foundation.operators import combine_systems, extract_system_couplings
from tachys.lattice.lattice_database import square
from tachys.lattice.operator.local_estimator import compute_expectation
from tachys.montecarlo import sample
from tachys.wavefunction import WaveFunction
from tachys.optimizer import MARCH, SR
from tachys.ground_state_training import train


def hubbard_square_pbc(L, t=1.0, U=0.0):
    """Hubbard model on an L×L square lattice with periodic boundary conditions.

    H = -t * sum_{<i,j>,σ} (c†_{i,σ} c_{j,σ} + h.c.) + U * sum_i n_{i,↑} n_{i,↓}

    Sites are indexed row-major: site(x, y) = x*L + y, with x in [0,L) and y in [0,L).
    """
    H = None

    for x in range(L):
        for y in range(L):
            i = x * L + y

            interaction = U * Nup(i) * Ndn(i)
            H = interaction if H is None else H + interaction

            for j in (x * L + (y + 1) % L, ((x + 1) % L) * L + y):
                hopping = (  - t * Cup_dag(i) * Cup(j)
                           + - t * Cup_dag(j) * Cup(i)
                           + - t * Cdn_dag(i) * Cdn(j)
                           + - t * Cdn_dag(j) * Cdn(i))
                H = H + hopping

    return H

L = 4
seed = 0
N = L * L
Ne = 10
N_steps  = 5
eta0     = 0.005

Us = [0, 2, 4, 8]
n_systems = len(Us)

N_mc_per_system = 8
N_mc = n_systems * N_mc_per_system

key = jax.random.key(seed)
lattice = square(shape=(L, L))

Hs = [hubbard_square_pbc(L, U=U) for U in Us]
H = combine_systems(Hs, N_mc_per_system)
system_couplings = extract_system_couplings(H)
system_ids = jnp.repeat(jnp.arange(n_systems), N_mc_per_system)

key, subkey = jax.random.split(key)
initial_occupations, *_ = init_config_spinful(key=jax.random.key(0), Ne=Ne, Ns=N, N_mc=N_mc)
state = FermionFoundationState(
    occupations=initial_occupations,
    lattice=lattice,
    Ne=Ne,
    system_couplings=system_couplings,
    system_ids=system_ids,
    n_systems=n_systems,
)

# Ansatz -- conditions its backflow correction on state.system_couplings, so
# the single shared network can distinguish which system a sample came from.
model = FermionFoundationRBM(hidden_units=64)

key, subkey = jax.random.split(key)
params = model.init(jax.random.key(1), state)
wf = WaveFunction(params=params, apply_fn=model.apply, dtype=jnp.float64)

# MC sampling
action = BondExchange.create(lattice, max_dist=1, Nbands=2)
lr_schedule = lambda step: eta0

optimizer = MARCH(diag_shift=1e-4, mode="real")
opt_state = optimizer.init(wf.params)

for step in range(N_steps):
        mc_keys = jax.random.split(jax.random.key(2), N_mc)
        state, log_amps, acceptance = sample(1, state, action, mc_keys, wf)
        E_L, e_mean, e2_mean = compute_expectation(H, wf, state, log_amps)
        lr = lr_schedule(step)
        updates, opt_state = optimizer(E_L, opt_state, state, wf)
        wf = wf.apply_gradients(updates, lr)
        print(e_mean)