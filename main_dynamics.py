
import jax
import jax.numpy as jnp
import numpy as np

from tachys.dynamics import TDVP, evolve
from tachys.ground_state_training import train
from tachys.lattice.ansatz.rbm import SpinRBM
from tachys.lattice.exact_diag import build_sparse_hamiltonian, spins_hilbert_space
from tachys.lattice.lattice_database import square
from tachys.lattice.operator.local_estimator import compute_expectation
from tachys.lattice.spins.hamiltonians.ising_transverse_field import (
    ising_transverse_field_square_pbc,
)
from tachys.lattice.spins.spin_action import SpinFlip
from tachys.lattice.spins.spin_operators import Sx
from tachys.lattice.spins.spin_state import SpinState, init_config_fixed_magn
from tachys.optimizer import MARCH
from tachys.parallel import MASTER, rank
from tachys.wavefunction import WaveFunction

L = 4
N = L * L
N_mc = 512
seed = 0

h0 = 2.0          # pre-quench transverse field (paramagnetic)
h1 = 1.0          # post-quench field
J = 1.0

N_opt_steps = 200
eta = 0.005

dt = 0.001
N_time_steps = 500

key = jax.random.key(seed)
lattice = square(shape=(L, L))

key, subkey = jax.random.split(key)
state = SpinState(spins=init_config_fixed_magn(subkey, N, sz=0, N_mc=N_mc), lattice=lattice)

model = SpinRBM(hidden_units=2 * N, dtype=jnp.float64, complex=True)
key, subkey = jax.random.split(key)
wf = WaveFunction(params=model.init(subkey, state), apply_fn=model.apply, dtype=jnp.float64)

action = SpinFlip()

# ─── 1. Ground state of H(h0) ─────────────────────────────────────────────────

H_initial = ising_transverse_field_square_pbc(L, J=J, h=h0)

key, subkey = jax.random.split(key)
key, state, wf, _opt_state, _history = train(
    subkey, H_initial, state, wf, MARCH(diag_shift=1e-4, mode="complex"), action,
    N_opt_steps, lambda step: eta, N_mc,
)

# ─── 2. Quench to H(h1) and evolve ────────────────────────────────────────────

H_quench = ising_transverse_field_square_pbc(L, J=J, h=h1)

# diag_shift=0.0: the eigenvalue truncation below is the regularizer. rcond caps
# the condition number of the retained subspace at 1/rcond
tdvp = TDVP(mode="complex", rcond=1e-8)

# Transverse magnetization m_x = (1/N) sum_i sigma^x_i, with sigma^x = 2 S^x.
Mx = (2.0 / N) * Sx(tuple(range(N)))
mx_vmc = []


def measure_mx(state, wf, step, ctx):
    """Measure m_x on stage 1's batch, so it costs no extra sampling.

    ``ctx`` carries the wavefunction, configurations and log-amplitudes from the
    start of the step, i.e. the ones that belong with ``ctx.t``.
    """
    mx = compute_expectation(Mx, ctx.wf, ctx.state, ctx.log_amps)[1]
    mx_vmc.append(float(mx.real))
    return {"mx": mx_vmc[-1]}


wf0 = wf   # keep the pre-quench state: the exact reference starts from it too

key, subkey = jax.random.split(key)
key, state, wf, _opt_state, history = evolve(
    subkey, H_quench, state, wf, tdvp, action,
    N_steps=N_time_steps, dt=dt, N_mc=N_mc,
    integrator="heun",
    tdvp_error_every=10,
    log_callback_fn=measure_mx,
)

# ─── 3. Exact reference, and the comparison plot ──────────────────────────────
# Propagating the *same* initial state exactly isolates the error of the
# variational dynamics from the error of the ground state it started from.

if rank == MASTER:
    import matplotlib.pyplot as plt
    from scipy.sparse.linalg import expm_multiply

    basis = spins_hilbert_space(N)
    full = SpinState(spins=jnp.asarray(basis), lattice=lattice)

    pack = lambda st: np.asarray(((np.asarray(st.spins) + 1) // 2) @ (2 ** np.arange(N)))
    Hmat, sorted_active = build_sparse_hamiltonian(full, H_quench, pack)
    order = np.searchsorted(sorted_active, pack(full))

    psi = np.zeros(len(basis), dtype=complex)
    psi[order] = np.asarray(jnp.exp(wf0.apply_fn(wf0.params, full)))
    psi /= np.linalg.norm(psi)

    flips = [np.arange(len(basis)) ^ (1 << i) for i in range(N)]
    mx_of = lambda v: sum(np.vdot(v, v[f]) for f in flips).real / N

    mx_ed = []
    for _ in range(N_time_steps):
        mx_ed.append(mx_of(psi))
        psi = expm_multiply(-1j * Hmat * dt, psi)

    plt.figure(figsize=(7, 4))
    plt.plot(history["t"], mx_ed, "k-", label="exact diagonalization")
    plt.plot(history["t"], mx_vmc, "o", ms=3, alpha=0.7, label="t-VMC")
    plt.xlabel("t")
    plt.ylabel(r"$m_x = \frac{1}{N}\sum_i \langle \sigma^x_i \rangle$")
    plt.title(rf"TFIM {L}$\times${L}: quench $h={h0} \to {h1}$")
    plt.legend()
    plt.tight_layout()
    plt.savefig("mx_dynamics.png", dpi=150)
    print(f"\nmax |m_x(t-VMC) - m_x(ED)| = {np.abs(np.array(mx_vmc) - np.array(mx_ed)).max():.3e}")
    print("wrote mx_dynamics.png")

# For a time-dependent quench instead of a sudden one, pass a callable. Build the
# operator once and rescale its couplings:
#
#     zz, sx = H_quench.operators
#     base = sx.coupling
#     ramp = lambda t: h0 + (h1 - h0) * min(t / T_ramp, 1.0)
#     H_of_t = lambda t: H_quench.replace(
#         operators=(zz, sx.replace(coupling=base * ramp(t) / h1)))
