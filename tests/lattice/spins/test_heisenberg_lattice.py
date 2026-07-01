import jax.numpy as jnp
import numpy as np
import pytest

from tachys.lattice.exact_diag import spins_hilbert_space, exact_diag
from tachys.lattice.spins.hamiltonians.heisenberg import heisenberg_hamiltonian
from tachys.lattice.spins.spin_state import SpinState
from tachys.lattice.lattice_database import square, triangular

# ---- reference values (J=1, PBC) ----------------------------------------
# 4x4 square: E0/N = -0.70178020...  (well-established benchmark)
SQUARE_4x4_E0_PER_SITE = -0.7017802005268037

# 4x4 triangular: E0/N = -0.534719682344766  (J=1, PBC, NN = a1 / a2 / a2-a1)
TRIANGULAR_4x4_E0_PER_SITE = -0.534719682344766

# 4x4 J1-J2 square: E0/N = -0.5286202094621775  (J1=1, J2=0.5, PBC,
# NN = a1 / a2, NNN = a1+a2 / a1-a2)
J1J2_SQUARE_4x4_E0_PER_SITE = -0.5286202094621775


def _pack(state):
    bits = (np.asarray(state.spins) + 1) // 2
    return (bits * 2 ** np.arange(state.spins.shape[-1])).sum(axis=-1)


def _run_ed(H, N):
    all_states = spins_hilbert_space(N)
    state = SpinState(spins=jnp.array(all_states, dtype=jnp.int8), Ns=N)
    eigenvalues, _ = exact_diag(state, H, _pack, k=1)
    return float(eigenvalues[0])


@pytest.fixture(scope="module")
def square_4x4_energy():
    lat = square(shape=(4, 4))
    H = heisenberg_hamiltonian(lat, nn=[((1, 0), 1.0), ((0, 1), 1.0)])
    return _run_ed(H, lat.Ns)


@pytest.fixture(scope="module")
def triangular_4x4_energy():
    lat = triangular(shape=(4, 4))
    H = heisenberg_hamiltonian(lat, nn=[((1, 0), 1.0), ((0, 1), 1.0), ((-1, 1), 1.0)])
    return _run_ed(H, lat.Ns)


@pytest.fixture(scope="module")
def j1j2_square_4x4_energy():
    lat = square(shape=(4, 4))
    J1, J2 = 1.0, 0.5
    H = heisenberg_hamiltonian(lat, nn=[
        ((1, 0), J1), ((0, 1), J1),
        ((1, 1), J2), ((1, -1), J2),
    ])
    return _run_ed(H, lat.Ns)


def test_square_4x4_ground_state_energy(square_4x4_energy):
    N = 4 * 4
    assert square_4x4_energy / N == pytest.approx(SQUARE_4x4_E0_PER_SITE, rel=1e-12)


def test_triangular_4x4_ground_state_energy(triangular_4x4_energy):
    N = 4 * 4
    assert triangular_4x4_energy / N == pytest.approx(TRIANGULAR_4x4_E0_PER_SITE, rel=1e-10)


def test_j1j2_square_4x4_ground_state_energy(j1j2_square_4x4_energy):
    N = 4 * 4
    assert j1j2_square_4x4_energy / N == pytest.approx(J1J2_SQUARE_4x4_E0_PER_SITE, rel=1e-10)
