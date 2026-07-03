from functools import partial

import jax
import jax.numpy as jnp
from tachys.lattice.ansatz.fermionic_transformer import FermionicTransformer
from tachys.lattice.bond_exchange import BondExchange
from tachys.lattice.fermions.fermion_state import FermionState, init_config_spinful
from tachys.lattice.fermions.hamiltonians.hubbard import hubbard_hamiltonian
from tachys.lattice.spins.hamiltonians.heisenberg import heisenberg_square_pbc
from tachys.lattice.spins.hamiltonians.ising_transverse_field import ising_transverse_field_square_pbc
from tachys.lattice.spins.spin_state import SpinState, init_config_fixed_magn
from tachys.lattice.spins.spin_action import SpinFlip
from tachys.lattice.lattice_database import square
from tachys.montecarlo import CompositeAction
from tachys.lattice.ansatz.spin_vit import SpinViT
from tachys.optimizer.optimizers import _build_ntk as _build_ntk_base
from tachys.parallel import mesh
from tachys.wavefunction import WaveFunction
from tachys.optimizer import SR, SPRING, MARCH
from tachys.ground_state_training import train

L = 4
seed = 0
N = L * L
Ne = 10
N_mc = 512
U = 8

key = jax.random.key(seed)
# Heisenberg Hamiltonian on a 4x4 square lattice with PBC
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
params = model.init(jax.random.key(1), state)
wf = WaveFunction(params=params, apply_fn=model.apply)

# MC sampling
action = BondExchange.create(lattice, max_dist=1, Nbands=2)

N_steps  = 1000
eta0     = 0.005
eta_final = 0.0005  # lr at step N_steps

def lr_schedule(step):
    """Linear decay from eta0 down to eta_final over N_steps."""
    frac = min(step / N_steps, 1.0)
    return eta0 + (eta_final - eta0) * frac

optimizer = MARCH(diag_shift=1e-4, mode="real")

use_wandb = False
wandb_run = None
if use_wandb:
    import wandb
    wandb_run = wandb.init(project="tachys", config={
        "L": L, "Ne": Ne, "N_mc": N_mc, "U": U,
        "N_steps": N_steps, "eta0": eta0, "eta_final": eta_final,
    })

key, state, wf, opt_state, history = train(
    key, H, state, wf, optimizer, action, N_steps, lr_schedule, N_mc, wandb_run=wandb_run
)

if wandb_run is not None:
    wandb_run.finish()