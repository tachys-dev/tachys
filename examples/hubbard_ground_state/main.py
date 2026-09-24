
import jax
import jax.numpy as jnp
from tachys.lattice.ansatz.fermionic_transformer import FermionicTransformer
from tachys.lattice.bond_exchange import BondExchange
from tachys.lattice.fermions.fermion_state import FermionState, init_config_spinful
from tachys.lattice.fermions.hamiltonians.hubbard import hubbard_hamiltonian
from tachys.lattice.lattice_database import square
from tachys.wavefunction import WaveFunction
from tachys.optimizer import MARCH
from tachys.ground_state_training import train

L = 4
seed = 0
N = L * L
Ne = 10
N_mc = 512
U = 8

N_steps  = 10
eta0     = 0.005

key = jax.random.key(seed)
lattice = square(shape=(L, L))
H = hubbard_hamiltonian(lattice, nn=[((1, 0), 1.0), ((0, 1), 1.0)], U=U)

key, subkey = jax.random.split(key)
initial_occupations, *_ = init_config_spinful(key=subkey, Ne=Ne, Ns=N, N_mc=N_mc)
state = FermionState(occupations=initial_occupations, lattice=lattice, Ne=Ne)

# RBM wavefunction
num_heads = 8
f = 4
model = FermionicTransformer(num_layers=2,
                 d_model=num_heads*f, 
                 num_heads=num_heads, 
                 Ns=N,
                 Ne=Ne,
                 two_dimensional=True,
                 transl_invariant=True,
                 dtype=jnp.float64)

# Initialise RBM parameters with a dummy forward pass
key, subkey = jax.random.split(key)
params = model.init(subkey, state)
wf = WaveFunction(params=params, apply_fn=model.apply, dtype=jnp.float64)

# MC sampling
action = BondExchange.create(lattice, max_dist=1, Nbands=2)

lr_schedule = lambda step: eta0

optimizer = MARCH(diag_shift=1e-4, mode="real")

key, subkey = jax.random.split(key)
key, state, wf, opt_state, history = train(
    subkey, H, state, wf, optimizer, action, N_steps, lr_schedule, N_mc
)