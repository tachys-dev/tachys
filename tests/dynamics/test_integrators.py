"""Heun and RK4, exercised on deterministic ODEs where the exact answer is known.

The right-hand side here is a plain function of ``(t, params)`` -- no sampling,
no wavefunction -- so these tests isolate the Butcher tableaux and the pytree
arithmetic from everything stochastic.
"""
import numpy as np
import pytest

import jax
import jax.numpy as jnp

from tachys.dynamics.integrators import ExplicitRK, Heun, RK4, get_integrator


class _FakeWF:
    """Minimal stand-in for WaveFunction: the integrator only ever touches
    ``.params`` and ``.replace(params=...)``."""

    def __init__(self, params):
        self.params = params

    def replace(self, params):
        return _FakeWF(params)


def _make_rhs(f, record=None):
    """Wrap ``f(t, params) -> dparams`` in the integrator's rhs protocol."""
    def rhs(key, t, wf, state):
        if record is not None:
            record.append((t, jax.tree.map(np.asarray, wf.params)))
        return key, state, f(t, wf.params), None
    return rhs


def _integrate(integrator, f, p0, dt, n_steps, t0=0.0, record=None):
    wf, state, key, t = _FakeWF(p0), None, jax.random.key(0), t0
    for _ in range(n_steps):
        key, wf, state, _ks, _aux = integrator.step(_make_rhs(f, record), key, t, wf, state, dt)
        t += dt
    return wf.params


@pytest.mark.parametrize("make,name,order,stages", [(Heun, "heun", 2, 2), (RK4, "rk4", 4, 4)])
def test_tableau_metadata(make, name, order, stages):
    integ = make()
    assert (integ.name, integ.order, integ.n_stages) == (name, order, stages)


def test_get_integrator_accepts_names_and_instances():
    assert get_integrator("RK4").name == "rk4"
    assert get_integrator("heun").name == "heun"
    inst = RK4()
    assert get_integrator(inst) is inst
    with pytest.raises(ValueError, match="Unknown integrator"):
        get_integrator("dopri5")


def test_tableau_is_validated():
    with pytest.raises(ValueError, match="strictly lower triangular"):
        ExplicitRK(name="bad", c=(0.0, 1.0), A=((), (1.0, 2.0)), b=(0.5, 0.5), order=2)
    with pytest.raises(ValueError, match="b must sum to 1"):
        ExplicitRK(name="bad", c=(0.0, 1.0), A=((), (1.0,)), b=(0.5, 0.9), order=2)
    with pytest.raises(ValueError, match="row-sum condition"):
        ExplicitRK(name="bad", c=(0.0, 0.5), A=((), (1.0,)), b=(0.5, 0.5), order=2)


@pytest.mark.parametrize("make", [Heun, RK4])
def test_constant_velocity_is_integrated_exactly(make):
    """dtheta/dt = c has the exact solution theta_0 + c t for any consistent scheme."""
    p0 = {"a": jnp.zeros(3), "b": jnp.asarray([1.0])}
    c = {"a": jnp.asarray([1.0, -2.0, 0.5]), "b": jnp.asarray([3.0])}
    out = _integrate(make(), lambda t, p: c, p0, dt=0.1, n_steps=10)
    assert np.allclose(np.asarray(out["a"]), [1.0, -2.0, 0.5], atol=1e-13)
    assert np.allclose(np.asarray(out["b"]), [1.0 + 3.0], atol=1e-13)


@pytest.mark.parametrize("make,order", [(Heun, 2), (RK4, 4)])
def test_convergence_order_on_a_linear_ode(make, order):
    """dtheta/dt = A theta, exact solution expm(A T) theta_0.

    Halving dt must divide the error by 2**order; a Butcher-coefficient typo
    that still satisfies the consistency conditions would show up here as a
    wrong exponent.
    """
    import scipy.linalg

    A = np.array([[0.0, 1.0, 0.0], [-2.0, 0.0, 0.5], [0.0, -1.0, 0.0]])
    p0 = jnp.asarray([1.0, 0.0, -0.5])
    T = 1.0
    exact = scipy.linalg.expm(A * T) @ np.asarray(p0)
    f = lambda t, p: jnp.asarray(A) @ p

    errs = []
    for n_steps in (20, 40, 80):
        got = _integrate(make(), f, p0, dt=T / n_steps, n_steps=n_steps)
        errs.append(np.linalg.norm(np.asarray(got) - exact))

    for coarse, fine in zip(errs, errs[1:]):
        assert coarse / fine == pytest.approx(2 ** order, rel=0.15), f"errors {errs}"


@pytest.mark.parametrize("make,order", [(Heun, 2), (RK4, 4)])
def test_convergence_order_on_an_explicitly_time_dependent_ode(make, order):
    """dtheta/dt = cos(t), exact solution sin(t) -- catches a wrong stage time
    ``c[i]``, which the autonomous test above cannot see."""
    p0 = jnp.zeros(1)
    T = 1.0
    f = lambda t, p: jnp.asarray([np.cos(t)])

    errs = []
    for n_steps in (10, 20, 40):
        got = _integrate(make(), f, p0, dt=T / n_steps, n_steps=n_steps)
        errs.append(abs(float(got[0]) - np.sin(T)))

    for coarse, fine in zip(errs, errs[1:]):
        assert coarse / fine == pytest.approx(2 ** order, rel=0.15), f"errors {errs}"


def test_rk4_stage_times_and_parameters_follow_the_tableau():
    """Stage i must be evaluated at t + c[i]*dt with theta_n + dt*sum_j A[i][j]*k_j."""
    record = []
    p0 = jnp.asarray([0.0])
    # constant velocity 1 => k_j = 1 for every stage, so the expected stage
    # parameters are simply dt * sum(A[i]) = dt * c[i]
    _integrate(RK4(), lambda t, p: jnp.asarray([1.0]), p0, dt=0.2, n_steps=1, t0=0.5, record=record)

    times = [t for t, _ in record]
    params = [float(p[0]) for _, p in record]
    assert times == pytest.approx([0.5, 0.6, 0.6, 0.7])
    assert params == pytest.approx([0.0, 0.1, 0.1, 0.2])


def test_step_returns_every_stage():
    """Both per-stage lists come back, so the driver can report stage-1 physics
    and reduce solver diagnostics over all stages."""
    rhs = lambda key, t, wf, state: (key, state, jnp.asarray([t]), f"aux@{t}")
    _key, wf, _state, ks, auxes = RK4().step(
        rhs, jax.random.key(0), 1.0, _FakeWF(jnp.zeros(1)), None, 0.4)
    assert len(ks) == len(auxes) == 4
    assert auxes == ["aux@1.0", "aux@1.2", "aux@1.2", "aux@1.4"]
    # theta_1 = 0 + dt * sum_i b_i * k_i, and here k_i is the stage time itself
    assert float(wf.params[0]) == pytest.approx(
        0.4 * (1.0 / 6 * 1.0 + 1 / 3 * 1.2 + 1 / 3 * 1.2 + 1 / 6 * 1.4))


def test_changing_dt_does_not_retrigger_tracing():
    """Tableau coefficients are passed as arrays, so a new dt rebinds a buffer
    instead of compiling a new kernel."""
    from tachys.dynamics.integrators import _tree_lincomb

    _tree_lincomb.clear_cache()
    base = {"a": jnp.zeros(4)}
    trees = [{"a": jnp.ones(4)}, {"a": jnp.ones(4)}]
    for dt in (0.1, 0.05, 0.025):
        _tree_lincomb(base, jnp.asarray([dt, 2 * dt]), trees)
    assert _tree_lincomb._cache_size() == 1
