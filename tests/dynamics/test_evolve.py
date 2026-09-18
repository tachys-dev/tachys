"""End-to-end real-time evolution, validated against exact quantum mechanics.

The central test is ``test_reproduces_exact_diagonalisation``: on two spins, the
ansatz ``log psi = a + b1 s1 + b2 s2 + c s1 s2`` with complex coefficients spans
the whole Hilbert space, so the TDVP projection is exact and t-VMC *must*
reproduce ``exp(-iHt)`` to within the integrator's discretization error. That
error is then measured and shown to fall at the scheme's order, which is what
tells the two apart.

Those runs use a frozen Markov chain holding every basis state, so they are
deterministic: with a complete ansatz the TDVP residual vanishes configuration by
configuration, which makes the velocity independent of how the batch is
distributed, but only if every configuration is actually present. (Sampling can
transiently drop one, which is genuine Monte Carlo error, not a defect --
``test_reproduces_exact_diagonalisation_when_sampling`` covers that path with a
correspondingly looser bound.)
"""
import contextlib
import io
from typing import Any

import numpy as np
import pytest
import scipy.linalg

import flax.linen as nn
import jax
import jax.numpy as jnp

from tachys.dynamics import TDVP, TDVPError, evolve
from tachys.lattice.ansatz.rbm import SpinRBM
from tachys.lattice.exact_diag import build_sparse_hamiltonian, spins_hilbert_space
from tachys.lattice.lattice_database import chain, square
from tachys.lattice.spins.hamiltonians.ising_transverse_field import (
    ising_transverse_field_square_pbc,
)
from tachys.lattice.spins.spin_action import SpinFlip
from tachys.lattice.spins.spin_operators import Sx, Sz
from tachys.lattice.spins.spin_state import SpinState, init_config_fixed_magn
from tachys.montecarlo import _BaseAction
from tachys.optimizer import SR
from tachys.wavefunction import WaveFunction


class FrozenAction(_BaseAction):
    """Proposes nothing: ``allowed_move`` is always False, so the configurations
    never change and the batch keeps whatever coverage it started with."""

    def __call__(self, key, state):
        return state, jnp.zeros(state.spins.shape[0], dtype=bool), 0.0


class Complete2Spin(nn.Module):
    """log psi(s) = a + b1 s1 + b2 s2 + c s1 s2, complex coefficients.

    Four complex numbers over a four-dimensional Hilbert space: a complete
    parameterization (of every state with no exactly vanishing amplitude), so
    TDVP is exact and only the integrator can be wrong.
    """

    @nn.compact
    def __call__(self, state):
        s = jnp.atleast_2d(state.spins).astype(jnp.float64)
        feats = jnp.stack([jnp.ones_like(s[:, 0]), s[:, 0], s[:, 1], s[:, 0] * s[:, 1]], -1)
        re = self.param('re', nn.initializers.normal(0.3), (4,), jnp.float64)
        im = self.param('im', nn.initializers.normal(0.3), (4,), jnp.float64)
        return feats @ (re + 1j * im)


class DiagonalPhase(nn.Module):
    """log psi(x) = -i theta V(x) with V(x) = -sum_<ij> s_i s_j."""
    src: Any
    dst: Any

    @nn.compact
    def __call__(self, state):
        theta = self.param('theta', lambda k: jnp.zeros((1,), jnp.float64))
        s = jnp.atleast_2d(state.spins)
        V = -jnp.sum(s[:, jnp.asarray(self.src)] * s[:, jnp.asarray(self.dst)], axis=-1)
        return -1j * theta[0] * V.astype(jnp.float64)


# ─── two-spin exact-diagonalisation fixture ───────────────────────────────────

_N2 = 2
_LAT2 = chain(_N2)
# open-chain TFIM on two sites: H = -sigma^z_0 sigma^z_1 - h (sigma^x_0 + sigma^x_1)
_H2 = -4.0 * Sz((0,)) * Sz((1,)) + (-2 * 0.7) * Sx((0, 1))
_BASIS2 = spins_hilbert_space(_N2)
_FULL2 = SpinState(spins=jnp.asarray(_BASIS2).astype(jnp.int8), lattice=_LAT2)


def _pack(state):
    return np.asarray(((np.asarray(state.spins) + 1) // 2) @ (2 ** np.arange(_N2)))


_HMAT2, _SORTED2 = build_sparse_hamiltonian(_FULL2, _H2, _pack)
_HDENSE2 = np.array(_HMAT2.todense(), copy=True)
# basis row -> matrix index, so amplitudes can be reordered into matrix order
_ORDER2 = np.searchsorted(_SORTED2, _pack(_FULL2))


def _amplitudes(wf):
    """Normalized amplitude vector in matrix order."""
    v = np.array(jnp.exp(wf.apply_fn(wf.params, _FULL2)), copy=True)
    out = np.empty_like(v)
    out[_ORDER2] = v
    return out / np.linalg.norm(out)


def _run_two_spin(dt, T, integrator, action=None, N_mc=None, seed=4, error_every=1):
    action = action if action is not None else FrozenAction()
    spins = jnp.asarray(_BASIS2).astype(jnp.int8)
    if N_mc is not None:
        spins = jnp.asarray(np.tile(_BASIS2, (N_mc // len(_BASIS2), 1))).astype(jnp.int8)
    state = SpinState(spins=spins, lattice=_LAT2)
    model = Complete2Spin()
    wf = WaveFunction(params=model.init(jax.random.key(5), state), apply_fn=model.apply)
    psi0 = _amplitudes(wf)

    with contextlib.redirect_stdout(io.StringIO()):
        _key, _state, wf_t, _opt, history = evolve(
            jax.random.key(seed), _H2, state, wf, TDVP(mode="complex", rcond=1e-10),
            action, N_steps=int(round(T / dt)), dt=dt, N_mc=spins.shape[0],
            integrator=integrator, tdvp_error_every=error_every,
        )

    exact = scipy.linalg.expm(-1j * _HDENSE2 * T) @ psi0
    exact = exact / np.linalg.norm(exact)
    infidelity = 1.0 - abs(np.vdot(exact, _amplitudes(wf_t))) ** 2
    return infidelity, history


def test_reproduces_exact_diagonalisation():
    """A complete ansatz must track exp(-iHt) to the integrator's accuracy."""
    infidelity, history = _run_two_spin(dt=0.0125, T=0.5, integrator="rk4")
    assert infidelity < 1e-13, f"1 - F = {infidelity:.3e}"
    # and the TDVP diagnostic agrees that there is no variational error
    assert max(history["tdvp_rate"]) < 1e-20
    assert history["R2"][-1] < 1e-12


@pytest.mark.parametrize("integrator,order,dts", [
    ("heun", 2, (0.05, 0.025, 0.0125)),
    # RK4 is already at 1e-14 by dt=0.025, so it has to be measured on a coarser
    # ladder or the fit runs into machine precision
    ("rk4", 4, (0.1, 0.05, 0.025)),
])
def test_integrator_order_against_exact_evolution(integrator, order, dts):
    """Infidelity ~ |state error|^2 ~ dt^(2*order), so halving dt divides it by
    4**order. A wrong Butcher coefficient changes that exponent."""
    T = 0.5
    errs = [_run_two_spin(dt, T, integrator, error_every=0)[0] for dt in dts]
    assert all(e > 1e-15 for e in errs), f"too close to machine precision to fit: {errs}"
    for coarse, fine in zip(errs, errs[1:]):
        assert coarse / fine == pytest.approx(4 ** order, rel=0.4), f"errors {errs}"


def test_reproduces_exact_diagonalisation_when_sampling():
    """Same check through the real sampler. Looser: a Metropolis batch can
    transiently miss a basis state, which under-determines the velocity on it."""
    infidelity, _ = _run_two_spin(dt=0.01, T=0.3, integrator="rk4",
                                  action=SpinFlip(), N_mc=64)
    assert infidelity < 1e-3, f"1 - F = {infidelity:.3e}"


# ─── exactly representable phase evolution, through the full loop ─────────────

_L, _NS, _NMC = 4, 16, 24


def _square_bonds(L):
    bonds = [(x * L + y, x * L + (y + 1) % L) for x in range(L) for y in range(L)]
    bonds += [(x * L + y, ((x + 1) % L) * L + y) for x in range(L) for y in range(L)]
    return zip(*bonds)


@pytest.mark.parametrize("integrator", ["heun", "rk4"])
def test_phase_ansatz_theta_equals_t(integrator):
    """theta(t) = t exactly -- the whole loop's sign and normalization."""
    src, dst = _square_bonds(_L)
    H = -4.0 * Sz(src) * Sz(dst)
    lattice = square(shape=(_L, _L))
    state = SpinState(spins=init_config_fixed_magn(jax.random.key(1), _NS, sz=0, N_mc=_NMC),
                      lattice=lattice)
    model = DiagonalPhase(src=src, dst=dst)
    wf = WaveFunction(params=model.init(jax.random.key(0), state), apply_fn=model.apply)

    with contextlib.redirect_stdout(io.StringIO()):
        _key, _state, wf_t, _opt, history = evolve(
            jax.random.key(3), H, state, wf, TDVP(mode="complex", rcond=1e-10),
            SpinFlip(), N_steps=5, dt=0.1, N_mc=_NMC, integrator=integrator,
            tdvp_error_every=1,
        )

    assert float(wf_t.params['params']['theta'][0]) == pytest.approx(0.5, abs=1e-12)
    assert history["R2"][-1] < 1e-12
    assert history["t"] == [0.0, 0.1, 0.2, 0.30000000000000004, 0.4]


# ─── driver plumbing ──────────────────────────────────────────────────────────

def _small_setup(hidden=2, N_mc=16, L=2):
    Ns = L * L
    lattice = square(shape=(L, L))
    state = SpinState(spins=init_config_fixed_magn(jax.random.key(1), Ns, sz=0, N_mc=N_mc),
                      lattice=lattice)
    model = SpinRBM(hidden_units=hidden, dtype=jnp.float64, complex=True)
    wf = WaveFunction(params=model.init(jax.random.key(0), state), apply_fn=model.apply)
    return state, wf, N_mc


def test_time_dependent_hamiltonian_is_accepted_and_does_not_recompile():
    """A coupling that varies with t is traced as *data*, so it costs nothing.

    Asserted as: a time-dependent run compiles exactly as many kernels as a
    static one, and adding steps adds none. (The absolute count is not 1 --
    parameters handed in by the caller and parameters produced by a jitted stage
    differ in device commitment, which buys one extra trace on the very first
    stage transition of any run, static or not.)
    """
    from tachys.lattice.operator.local_estimator import compute_expectation

    H0 = ising_transverse_field_square_pbc(2, J=1.0, h=1.0)
    zz, sx = H0.operators
    base = sx.coupling

    def H_of_t(t):
        return H0.replace(operators=(zz, sx.replace(coupling=base * (1.0 + 2.0 * t))))

    def cache_after(H, n_steps):
        state, wf, N_mc = _small_setup()
        compute_expectation.clear_cache()
        with contextlib.redirect_stdout(io.StringIO()):
            _k, _s, _w, _o, history = evolve(
                jax.random.key(3), H, state, wf, TDVP(mode="complex"), SpinFlip(),
                N_steps=n_steps, dt=0.01, N_mc=N_mc, integrator="heun")
        assert len(history["t"]) == n_steps
        return compute_expectation._cache_size()

    static = cache_after(H0, 3)
    assert cache_after(H_of_t, 3) == static
    assert cache_after(H_of_t, 8) == static


def test_time_dependent_hamiltonian_structure_is_validated():
    """A H(t) whose pytree/avals change with t would retrace every stage; say so
    up front instead of silently crawling."""
    state, wf, N_mc = _small_setup()
    H0 = ising_transverse_field_square_pbc(2, J=1.0, h=1.0)
    zz, sx = H0.operators

    def bad_H(t):
        # coupling grows an extra entry -> different leaf shape -> new jit key
        n = 1 if t == 0.0 else 2
        return H0.replace(operators=(zz, sx.replace(coupling=jnp.ones(n))))

    with pytest.raises(ValueError, match="same operator structure"):
        with contextlib.redirect_stdout(io.StringIO()):
            evolve(jax.random.key(3), bad_H, state, wf, TDVP(mode="complex"), SpinFlip(),
                   N_steps=1, dt=0.01, N_mc=N_mc)

    with pytest.raises(TypeError, match="not a tachys operator"):
        with contextlib.redirect_stdout(io.StringIO()):
            evolve(jax.random.key(3), lambda t: 1.0, state, wf, TDVP(mode="complex"),
                   SpinFlip(), N_steps=1, dt=0.01, N_mc=N_mc)


def test_static_hamiltonian_is_not_mistaken_for_a_callable():
    """Every tachys operator defines __call__(state), so the dispatch must test
    isinstance first -- a plain callable() check would call H(t) with a float."""
    state, wf, N_mc = _small_setup()
    H = ising_transverse_field_square_pbc(2, J=1.0, h=1.0)
    with contextlib.redirect_stdout(io.StringIO()):
        _key, _state, _wf, _opt, history = evolve(
            jax.random.key(3), H, state, wf, TDVP(mode="complex"), SpinFlip(),
            N_steps=2, dt=0.01, N_mc=N_mc, integrator="heun")
    assert len(history["energy"]) == 2


def test_callbacks_of_both_arities_are_supported():
    """train()'s cb(state, wf, step) still works; cb(state, wf, step, ctx) gets
    the time, the step's Hamiltonian, the stage-1 batch and the velocity."""
    state, wf, N_mc = _small_setup()
    H = ising_transverse_field_square_pbc(2, J=1.0, h=1.0)

    seen_three, seen_four = [], []

    def three_arg(state, wf, step):
        seen_three.append(step)
        return {"three": step}

    def four_arg(state, wf, step, ctx):
        seen_four.append((step, ctx.t))
        assert ctx.dt == 0.02
        assert ctx.Ns == 4
        assert ctx.mode == "complex"
        assert ctx.E_L.shape[0] == N_mc
        assert jax.tree.structure(ctx.dtheta_dt) == jax.tree.structure(wf.params)
        assert len(ctx.stages) == 2               # heun
        return None                                # None results are skipped

    with contextlib.redirect_stdout(io.StringIO()):
        evolve(jax.random.key(3), H, state, wf, TDVP(mode="complex"), SpinFlip(),
               N_steps=3, dt=0.02, N_mc=N_mc, integrator="heun",
               log_callback_fn=[three_arg, four_arg])

    assert seen_three == [0, 1, 2]
    assert [s for s, _ in seen_four] == [0, 1, 2]
    assert [round(t, 6) for _, t in seen_four] == [0.0, 0.02, 0.04]


def test_tdvp_error_callback_can_be_registered_manually():
    """TDVPError works through log_callback_fn too, not only via
    tdvp_error_every."""
    state, wf, N_mc = _small_setup()
    H = ising_transverse_field_square_pbc(2, J=1.0, h=1.0)
    cb = TDVPError(every=2)

    with contextlib.redirect_stdout(io.StringIO()):
        evolve(jax.random.key(3), H, state, wf, TDVP(mode="complex"), SpinFlip(),
               N_steps=5, dt=0.01, N_mc=N_mc, integrator="heun", log_callback_fn=cb)

    assert cb.history["step"] == [0, 2, 4]
    assert cb.R2 > 0.0


def test_start_step_and_t0_offset_the_run():
    state, wf, N_mc = _small_setup()
    H = ising_transverse_field_square_pbc(2, J=1.0, h=1.0)
    with contextlib.redirect_stdout(io.StringIO()):
        _key, _state, _wf, _opt, history = evolve(
            jax.random.key(3), H, state, wf, TDVP(mode="complex"), SpinFlip(),
            N_steps=3, dt=0.5, N_mc=N_mc, integrator="heun", t0=2.0, start_step=10)
    assert history["t"] == [2.0, 2.5, 3.0]


def test_evolve_rejects_a_ground_state_optimizer():
    """SR solves the imaginary-time equation; handing it to evolve would quietly
    relax towards the ground state instead of propagating."""
    state, wf, N_mc = _small_setup()
    H = ising_transverse_field_square_pbc(2, J=1.0, h=1.0)
    with pytest.raises(TypeError, match="must be a tachys.dynamics.TDVP"):
        evolve(jax.random.key(3), H, state, wf, SR(diag_shift=1e-4, mode="complex"),
               SpinFlip(), N_steps=1, dt=0.01, N_mc=N_mc)


def test_evolve_rejects_a_chain_count_mismatch():
    state, wf, N_mc = _small_setup()
    H = ising_transverse_field_square_pbc(2, J=1.0, h=1.0)
    with pytest.raises(ValueError, match="chains but N_mc"):
        evolve(jax.random.key(3), H, state, wf, TDVP(mode="complex"), SpinFlip(),
               N_steps=1, dt=0.01, N_mc=N_mc + 8)
