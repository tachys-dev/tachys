from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from jax import shard_map
from jax.sharding import PartitionSpec as P

from tachys.lattice.foundation.collectives import grouped_mean
from tachys.lattice.foundation.foundation_state import FoundationState
from tachys.optimizer.optimizers import _make_apply_fn
from tachys.parallel import mesh, n_devices
from tachys.utils import _cast_floating_to


class TDVPErrorEstimate(struct.PyTreeNode):
    """One measurement of the TDVP residual. All fields are global scalars.

    Attributes
    ----------
    rate  : ``delta s^2 / delta t^2``, from the fused form -- non-negative exactly.
    var_H : ``Var(H) = mean |DeltaE_L|^2``. The ``thetadot = 0`` baseline: the rate
            a frozen ansatz would accumulate.
    quad  : ``thetadot^T S thetadot = mean |t|^2``.
    force : ``2 Im(F)^T thetadot = 2 Im mean[conj(t) DeltaE_L]``, with
            ``F_k = mean[conj(DeltaO_k) DeltaE_L]``.
    ratio : ``force / (2 * quad)``. Exactly 1 when ``thetadot`` solves the TDVP
            equation, so it is a direct check on the *normalization* of the
            velocity: 0.5 means ``thetadot`` is twice too large, 2.0 twice too
            small, -1 means its sign is flipped. Deviations also grow with the
            eigenvalue truncation, which is legitimate -- that is the
            regularization moving ``thetadot`` off the exact minimizer.
    decomposed : ``var_H + quad - force``, algebraically identical to ``rate``.
            Their relative difference is a free running check on floating-point
            cancellation and on batch consistency.
    """
    rate: jnp.ndarray
    var_H: jnp.ndarray
    quad: jnp.ndarray
    force: jnp.ndarray
    ratio: jnp.ndarray
    decomposed: jnp.ndarray


def _global_mean(x):
    """Mean over the MC axis of *all* devices, from inside a shard_map."""
    return jax.lax.psum(jnp.mean(x), 'i') / n_devices


def _center(x, state):
    """Subtract the mean the same way ``optimizers._center_eloc`` does."""
    if isinstance(state, FoundationState):
        return x - grouped_mean(x, state.system_ids, state.n_systems, broadcast=True)
    return x - _global_mean(x)


@partial(jax.jit, static_argnames=("mode",))
def tdvp_error_rate(wf, state, E_L, dtheta_dt, mode="complex"):
    """``delta s^2 / delta t^2`` and its decomposition, from a single JVP.

    Parameters
    ----------
    wf        : WaveFunction — must hold the parameters ``E_L`` was measured at,
                i.e. the ones *before* the integrator step. Taking the JVP at the
                advanced parameters would introduce an O(dt) inconsistency into
                exactly the small residual being measured.
    state     : State — the batch ``E_L`` was measured on, sharded on 'i'.
    E_L       : jax.Array (N_mc_local,) — local energies on that batch, sharded on 'i'.
    dtheta_dt : pytree matching ``wf.params`` — the velocity to evaluate the
                residual for. Real, same structure/shapes/dtype as the params.
    mode      : "complex" or "real", as for the optimizers.

    Returns
    -------
    TDVPErrorEstimate

    Notes
    -----
    ``Var(H)`` is ``mean|E_L|^2 - |mean E_L|^2`` -- the *modulus*, not the real
    part. ``compute_expectation`` returns both moments in that convention, but
    ``ground_state_training._compute_metrics`` takes real parts first, which is
    fine at a converged ground state and wrong here.

    All three terms must come from the same batch: that is what makes the fused
    form exactly non-negative.
    """
    def _body(wf, state, E_L, dtheta_dt):
        wf, state = _cast_floating_to((wf, state), wf.dtype)
        apply_fn = _make_apply_fn(wf.apply_fn, mode)

        # t_i = sum_k DeltaO_ik thetadot_k, forward mode: no n_params-sized object.
        _, Jdot = jax.jvp(lambda p: apply_fn(p, state), (wf.params,), (dtheta_dt,))
        if mode == "complex":
            # +1j. optimizers._jvp_correction uses -1j because the object it
            # builds lives in the conjugate ("eps") space; copying that sign here
            # flips the cross term, turning the rate into var + quad + force --
            # still non-negative, still plausible-looking, and wrong.
            t = jax.lax.complex(Jdot[:, 0], Jdot[:, 1])
        else:
            t = Jdot.astype(jnp.complex128)

        t  = _center(t, state)
        dE = _center(E_L.astype(jnp.complex128), state)

        quad  = jnp.real(_global_mean(jnp.abs(t) ** 2))
        force = 2.0 * jnp.imag(_global_mean(jnp.conj(t) * dE))
        var_H = jnp.real(_global_mean(jnp.abs(dE) ** 2))
        rate  = jnp.real(_global_mean(jnp.abs(t + 1j * dE) ** 2))

        return TDVPErrorEstimate(
            rate=rate,
            var_H=var_H,
            quad=quad,
            force=force,
            ratio=force / jnp.where(quad > 0, 2.0 * quad, jnp.inf),
            decomposed=var_H + quad - force,
        )

    return shard_map(
        _body, mesh=mesh,
        in_specs=(P(), P('i'), P('i'), P()),
        out_specs=P(),
        check_vma=False,
    )(wf, state, E_L, dtheta_dt)


class TDVPError:
    """Accumulates the integrated TDVP error ``R^2`` from ``tdvp_error_rate``
    measurements.

    ``evolve(..., tdvp_error_every=10)`` builds one and feeds it every 10 steps.
    The measurement uses the **first stage** of the step: the velocity ``k_1``,
    the batch and the local energies all evaluated at ``(t_n, theta_n)``, the only
    stage that sits on the trajectory and the only one whose three terms share a
    batch. ``k_1`` is the TDVP velocity proper; the residual is a statement about
    the variational manifold at ``t_n``, not about the integrator, so the
    multi-stage average is not the right object here.

    Parameters
    ----------
    rule  : ``"rect"`` (default) or ``"trapezoid"``. With measurements every
            ``n > 1`` steps each one has to stand in for the steps between
            measurements. ``"rect"`` is the literal reading of (12) -- a
            left-endpoint Riemann sum of width ``n * dt``; ``"trapezoid"``
            averages consecutive measurements over the same interval, which
            costs nothing and is strictly more accurate, but is not what the
            definition says. Both coincide as ``n -> 1``.
    prefix : str — key prefix for the returned metrics dict.

    Attributes
    ----------
    R2      : float — the accumulated error at the last measured step.
    history : dict[str, list] — per-measurement ``step``, ``t``, ``rate``,
              ``R2``, ``var_H``, ``quad``, ``force``, ``ratio``.
    """

    RULES = ("rect", "trapezoid")

    def __init__(self, rule="rect", prefix="tdvp"):
        if rule not in self.RULES:
            raise ValueError(f"rule must be one of {self.RULES}, got {rule!r}")
        self.rule = rule
        self.prefix = prefix
        self.reset()

    def reset(self):
        self.R2 = 0.0
        self._prev_sqrt_rate = None
        self._prev_t = None
        self.history = {k: [] for k in
                        ("step", "t", "rate", "R2", "var_H", "quad", "force", "ratio")}

    def accumulate(self, step, t, Ns, est):
        """Fold one ``TDVPErrorEstimate`` into ``R^2``; returns the metrics to log."""
        # jnp.maximum guards only against a -1e-19 from rounding; the fused
        # estimator cannot produce a genuinely negative rate.
        s = float(jnp.sqrt(jnp.maximum(est.rate, 0.0)))

        if self._prev_sqrt_rate is not None:
            # Width of the interval this measurement stands for -- computed from
            # the elapsed time rather than from `every * dt`, so a changed
            # `every`, a skipped measurement or a short final block all work.
            width = t - self._prev_t
            incr = (width * self._prev_sqrt_rate if self.rule == "rect"
                    else 0.5 * width * (self._prev_sqrt_rate + s))
            self.R2 += incr / Ns ** 0.5
        self._prev_sqrt_rate, self._prev_t = s, t

        rate, var_H = float(est.rate), float(est.var_H)
        for k, v in (("step", step), ("t", t), ("rate", rate), ("R2", self.R2),
                     ("var_H", var_H), ("quad", float(est.quad)),
                     ("force", float(est.force)), ("ratio", float(est.ratio))):
            self.history[k].append(v)

        p = self.prefix
        metrics = {
            f"{p}/R2": self.R2,
            f"{p}/rate": rate,
            f"{p}/var_H": var_H,
            f"{p}/ratio": float(est.ratio),
            # rate / Var(H) in [0, 1]: the fraction of the exact evolution
            # direction the manifold fails to capture. 0 = exact, 1 = the ansatz
            # is doing nothing a frozen state would not do.
            f"{p}/rel": rate / var_H if var_H > 0 else float("nan"),
        }
        # Free consistency check: `decomposed` is algebraically `rate`, so a
        # large disagreement means cancellation or a batch mismatch upstream.
        dec = float(est.decomposed)
        if rate > 0 and abs(dec - rate) > 1e-6 * max(abs(dec), rate, var_H):
            metrics[f"{p}/decomposed"] = dec
        return metrics


def blocking_error(x, min_blocks=8):
    """Error bar on the mean of a correlated series, by binning analysis.

    Re-exported from ``tachys.experimental.fidelity`` for convenience when
    error-barring the per-step ``rate`` series (which is autocorrelated exactly
    like the energy).
    """
    from tachys.experimental.fidelity import blocking_error as _be
    return _be(np.asarray(x), min_blocks=min_blocks)
