import dataclasses

import jax
import jax.numpy as jnp


@jax.jit
def _tree_lincomb(base, coeffs, trees):
    """``base + sum_i coeffs[i] * trees[i]``, leaf by leaf.

    ``coeffs`` is a 1-D array (not a list of Python floats) so that changing
    ``dt`` -- an adaptive or ramped step, or simply a second run -- rebinds a
    buffer instead of triggering a recompilation.

    This is deliberately *not* ``WaveFunction.apply_gradients``. That helper
    computes ``p - eta * g`` from a single tree, which (a) cannot express an
    ``s``-term combination at all and (b) reads as gradient *descent* everywhere
    else in the codebase; calling it with a negative ``eta`` at exactly the place
    where a sign error produces a plausible-looking time-reversed trajectory is
    the wrong trade.
    """
    return jax.tree.map(
        lambda b, *ks: b + sum(c * k for c, k in zip(coeffs, ks)),
        base, *trees,
    )


@dataclasses.dataclass(frozen=True)
class ExplicitRK:
    """An explicit Runge-Kutta scheme, defined by its Butcher tableau.

    Attributes
    ----------
    name  : str        — short identifier used in logs and the setup summary.
    c     : tuple      — stage times, ``t + c[i] * dt``. ``len(c) == n_stages``.
    A     : tuple      — strictly lower-triangular coefficient rows; ``A[i]`` has
                         length ``i`` and gives stage ``i``'s parameters as
                         ``theta_n + dt * sum_j A[i][j] * k_j``.
    b     : tuple      — quadrature weights; ``theta_{n+1} = theta_n + dt * sum_i b[i] * k_i``.
    order : int        — classical order of convergence of the deterministic scheme.
    """

    name: str
    c: tuple
    A: tuple
    b: tuple
    order: int

    def __post_init__(self):
        s = len(self.b)
        if len(self.c) != s or len(self.A) != s:
            raise ValueError(f"{self.name}: c, A and b must all have {s} entries")
        for i, row in enumerate(self.A):
            if len(row) != i:
                raise ValueError(
                    f"{self.name}: A[{i}] has {len(row)} entries, expected {i} "
                    "(an explicit scheme's tableau is strictly lower triangular)"
                )
        if abs(sum(self.b) - 1.0) > 1e-12:
            raise ValueError(f"{self.name}: b must sum to 1, got {sum(self.b)}")
        for i, (ci, row) in enumerate(zip(self.c, self.A)):
            if abs(sum(row) - ci) > 1e-12:
                raise ValueError(
                    f"{self.name}: row-sum condition violated at stage {i}: "
                    f"sum(A[{i}]) = {sum(row)} != c[{i}] = {ci}"
                )

    @property
    def n_stages(self):
        return len(self.b)

    def step(self, rhs, key, t, wf, state, dt):
        """Advance the parameters by one step of size ``dt``.

        Parameters
        ----------
        rhs   : callable ``(key, t, wf, state) -> (key, state, thetadot, aux)``.
                ``thetadot`` is the physical velocity ``dtheta/dt``, a pytree
                matching ``wf.params``; ``aux`` is opaque to the integrator and
                passed straight back to the caller.
        key   : jax.random.key — threaded through every stage.
        t     : float — time at the start of the step.
        wf    : WaveFunction at ``t``.
        state : State — Markov chain configurations, carried through the stages.
        dt    : float — step size.

        Returns
        -------
        key, wf, state, ks, auxes
            ``wf`` holds the advanced parameters, ``state`` the configurations
            left by the *last* stage (the best-equilibrated ones available: that
            stage sits within O(dt^2) of ``theta_{n+1}``, while stage 1 is O(dt)
            away). ``ks`` and ``auxes`` are per-stage lists, so the caller can
            report stage-1 physics and reduce solver-health diagnostics over all
            stages.
        """
        params0 = wf.params
        ks, auxes = [], []

        for i in range(self.n_stages):
            if i == 0:
                wf_i = wf
            else:
                # Drop the zero coefficients of the tableau (RK4's A[2][0],
                # A[3][0], A[3][1]) so no pointless multiply-add is emitted.
                idx = [j for j, a in enumerate(self.A[i]) if a != 0.0]
                coeffs = jnp.asarray([dt * self.A[i][j] for j in idx])
                params_i = _tree_lincomb(params0, coeffs, [ks[j] for j in idx])
                wf_i = wf.replace(params=params_i)

            key, state, k, aux = rhs(key, t + self.c[i] * dt, wf_i, state)
            ks.append(k)
            auxes.append(aux)

        new_params = _tree_lincomb(params0, jnp.asarray([dt * b for b in self.b]), ks)
        return key, wf.replace(params=new_params), state, ks, auxes


def Heun():
    """Explicit trapezoidal rule (Heun's method), 2nd order, 2 stages.

    ::

        0 |
        1 | 1
        --+---------
          | 1/2  1/2

        k1 = f(t,      theta_n)
        k2 = f(t + dt, theta_n + dt * k1)
        theta_{n+1} = theta_n + dt/2 * (k1 + k2)
    """
    return ExplicitRK(name="heun", c=(0.0, 1.0), A=((), (1.0,)), b=(0.5, 0.5), order=2)


def RK4():
    """Classical Runge-Kutta, 4th order, 4 stages.

    ::

        0   |
        1/2 | 1/2
        1/2 | 0    1/2
        1   | 0    0    1
        ----+--------------------
            | 1/6  1/3  1/3  1/6
    """
    return ExplicitRK(
        name="rk4",
        c=(0.0, 0.5, 0.5, 1.0),
        A=((), (0.5,), (0.0, 0.5), (0.0, 0.0, 1.0)),
        b=(1 / 6, 1 / 3, 1 / 3, 1 / 6),
        order=4,
    )


_INTEGRATORS = {"heun": Heun, "rk4": RK4}


def get_integrator(integrator):
    """Resolve ``"heun"`` / ``"rk4"`` (or an ``ExplicitRK`` instance) to a scheme."""
    if isinstance(integrator, ExplicitRK):
        return integrator
    try:
        return _INTEGRATORS[str(integrator).lower()]()
    except KeyError:
        raise ValueError(
            f"Unknown integrator {integrator!r}; expected one of "
            f"{sorted(_INTEGRATORS)} or an ExplicitRK instance."
        ) from None
