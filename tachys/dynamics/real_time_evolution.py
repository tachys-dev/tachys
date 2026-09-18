import dataclasses
import inspect
import sys
import time
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

from tachys.checkpoint import (
    build_checkpoint_manager,
    resolve_checkpoint_settings,
    save_training_checkpoint,
)
from tachys.dynamics.error import TDVPError
from tachys.dynamics.integrators import get_integrator
from tachys.dynamics.tdvp import TDVP
from tachys.ground_state_training import Timer, _format_fields
from tachys.lattice.operator.base import _OperatorBase
from tachys.lattice.operator.local_estimator import compute_expectation
from tachys.lattice.state_array import get_n_mc_local
from tachys.montecarlo import sample
from tachys.parallel import MASTER, n_devices, rank


# ─── Time-dependent Hamiltonians ──────────────────────────────────────────────

def _as_hamiltonian_fn(H):
    """Normalize ``H`` to a callable ``t -> operator``; also say whether it varies.

    The ``isinstance`` test must come first: every tachys operator defines
    ``__call__(state)``, so a plain ``callable(H)`` check would classify a static
    Hamiltonian as time dependent and then call it with a float, failing deep
    inside ``apply`` with an unrelated-looking AttributeError.
    """
    if isinstance(H, _OperatorBase):
        return (lambda t: H), False
    if not callable(H):
        raise TypeError(
            f"H must be a tachys operator or a callable t -> operator, got "
            f"{type(H).__name__}."
        )
    return H, True


def _aval_signature(H):
    """(shape, dtype, weak_type) of every leaf — the part of jit's cache key that
    a time-dependent Hamiltonian must hold fixed. ``utils.same_treedef_and_avals``
    is not enough here: it ignores ``weak_type``, which jit does not."""
    return [(jnp.shape(l), jnp.result_type(l), bool(getattr(jnp.asarray(l).aval, "weak_type", False)))
            for l in jax.tree.leaves(H)]


def _check_hamiltonian_fn(H_fn, t0, dt):
    """Fail early and legibly if ``H(t)`` changes anything jit keys on.

    Only the *values* of an operator's ``coupling`` may depend on ``t``: those are
    ordinary pytree leaves, so ``compute_expectation`` traces them as data and is
    not recompiled when they change. Anything else -- a different pytree
    structure, a different static field, a different leaf shape/dtype -- is part
    of the compilation key and would retrace every stage, turning a cheap ramp
    into a recompile storm.
    """
    H0, H1 = H_fn(t0), H_fn(t0 + dt)
    for Ht, t in ((H0, t0), (H1, t0 + dt)):
        if not isinstance(Ht, _OperatorBase):
            raise TypeError(f"H({t}) returned {type(Ht).__name__}, not a tachys operator.")
    if (repr(jax.tree.structure(H0)) != repr(jax.tree.structure(H1))
            or _aval_signature(H0) != _aval_signature(H1)):
        raise ValueError(
            "H(t) must return the same operator structure at every time: identical "
            "pytree structure, identical static fields, identical leaf "
            "shape/dtype/weak_type. Only the numerical values of `coupling` may "
            "depend on t.\n"
            "Build the operator once and return a copy with rescaled couplings, e.g.\n"
            "    H0 = ising_transverse_field_square_pbc(L, J=1.0, h=1.0)\n"
            "    zz, sx = H0.operators\n"
            "    H = lambda t: H0.replace(operators=(zz, sx.replace(coupling=sx.coupling * h(t))))\n"
            "Do not call the lattice Hamiltonian builder inside H(t) (it is orders of "
            "magnitude slower), and do not write `f(t) * op` with a jax scalar "
            "(_Operator.__mul__ only accepts Python scalars)."
        )
    return H0


# ─── Per-stage bookkeeping ────────────────────────────────────────────────────

@dataclasses.dataclass
class StageAux:
    """Everything one Runge-Kutta stage produced, for diagnostics.

    ``state``/``log_amps``/``E_L`` are references to arrays the stage already
    built, not copies, so keeping them costs nothing and lets the TDVP-error
    callback reuse stage 1's batch instead of drawing its own.
    """
    t: float
    state: Any
    log_amps: Any
    E_L: Any
    e_mean: Any
    e2_mean: Any
    acceptance: Any
    t_mc: float
    t_expect: float
    t_solve: float


@dataclasses.dataclass
class DynamicsContext:
    """Argument passed to 4-argument dynamics callbacks.

    All of it describes the step that has just been taken, evaluated at its
    *start*: ``wf``/``state``/``E_L``/``dtheta_dt`` are the stage-1 quantities at
    ``(t, theta_n)``, the only mutually consistent set (same parameters, same
    batch, same Hamiltonian). That is what the TDVP error needs -- taking the JVP
    at the post-step parameters would put an O(dt) inconsistency straight into
    the small residual being measured.
    """
    step: int
    t: float
    dt: float
    H: Any
    wf: Any               # WaveFunction BEFORE the step
    state: Any            # stage-1 Monte Carlo batch
    log_amps: Any
    E_L: Any
    e_mean: Any
    e2_mean: Any
    dtheta_dt: Any        # stage-1 velocity k1
    acceptance: Any
    mode: str
    Ns: int
    stages: tuple         # tuple[StageAux], one per RK stage


class _TDVPRhs:
    """The right-hand side handed to the integrator: one sample + solve.

    ``(key, t, wf, state) -> (key, state, dtheta_dt, aux)``. ``wf`` carries the
    stage's parameters and ``state`` the Markov chain inherited from the previous
    stage (warm start); both are passed on rather than reconstructed, which is
    what keeps the jitted callees from retracing between stages.
    """

    def __init__(self, H_fn, tdvp, opt_state, action, N_mc, nsweeps):
        self.H_fn = H_fn
        self.tdvp = tdvp
        self.opt_state = opt_state
        self.action = action
        self.N_mc = N_mc
        self.nsweeps = nsweeps

    def __call__(self, key, t, wf, state):
        key, subkey = jax.random.split(key)

        with Timer() as t_mc:
            mc_keys = jax.random.split(subkey, self.N_mc)
            state, log_amps, acceptance = sample(self.nsweeps, state, self.action, mc_keys, wf)
            jax.block_until_ready((state, log_amps))

        with Timer() as t_expect:
            # H at THIS stage's time: an RK4 stage sits at t + dt/2 or t + dt, and
            # using the step's own H there would silently lower the order.
            E_L, e_mean, e2_mean = compute_expectation(self.H_fn(t), wf, state, log_amps)
            jax.block_until_ready(E_L)

        with Timer() as t_solve:
            dtheta_dt, self.opt_state = self.tdvp(E_L, self.opt_state, state, wf)
            jax.block_until_ready(dtheta_dt)

        aux = StageAux(t=t, state=state, log_amps=log_amps, E_L=E_L, e_mean=e_mean,
                       e2_mean=e2_mean, acceptance=acceptance, t_mc=t_mc.elapsed,
                       t_expect=t_expect.elapsed, t_solve=t_solve.elapsed)
        return key, state, dtheta_dt, aux


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _callback_takes_ctx(cb):
    """True if ``cb`` accepts a 4th positional argument (the DynamicsContext).

    Lets ``evolve`` accept both the dynamics protocol ``cb(state, wf, step, ctx)``
    and ``train``'s 3-argument ``cb(state, wf, step)``, so observable-measuring
    callbacks written for ground-state runs work here unchanged.
    """
    try:
        params = inspect.signature(cb).parameters.values()
    except (TypeError, ValueError):
        return False
    n = 0
    for p in params:
        if p.kind is p.VAR_POSITIONAL:
            return True
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
            n += 1
    return n >= 4


def _dynamics_metrics(e_mean, e2_mean, Ns):
    """Energy per site and variance per site, in the complex-E_L convention.

    ``Var(H) = <|E_L|^2> - |<E_L>|^2`` uses the modulus of the mean, unlike
    ``ground_state_training._compute_metrics``, which takes real parts first.
    At a converged real ground state the difference is nil; in real time
    ``Im <E_L>`` is only zero up to Monte Carlo noise and dropping it would
    inflate the variance (and, through it, the TDVP error).
    """
    e = complex(e_mean)
    var = float(jnp.real(e2_mean)) - abs(e) ** 2
    return e.real, e.real / Ns, var / Ns, var


def _check_finite(e_per_site, t, wandb_run):
    """Abort the run if the energy has gone non-finite.

    Unlike ``train``, there is no sensible fixed window here: the energy is a
    conserved quantity of the dynamics, not something being driven down, and its
    physical value depends entirely on the initial state and the quench. A
    diverging trajectory shows up first as NaN (from the solve or the sampler),
    which every rank sees identically, so each independently reaches the same
    verdict and exits without needing a broadcast.
    """
    if not np.isfinite(e_per_site):
        print(f"Energy is not finite (E/N = {e_per_site}) at t = {t:.6g}; aborting.", flush=True)
        if wandb_run is not None:
            wandb_run.tags += ("divergence",)
            wandb_run.finish()
        sys.exit(1)


def _print_setup_summary(H, wf, tdvp, integrator, action, state, N_steps, dt, N_mc,
                         t0, nsweeps, time_dependent, tdvp_error_every, start_step):
    lattice = state.lattice
    model = getattr(wf.apply_fn, "__self__", None)

    print("\n--- Real-time evolution setup ---")
    if lattice is not None:
        print(f"Lattice      : {type(lattice).__name__}  Ns={lattice.Ns}  L={lattice.L}  "
              f"nb={lattice.nb}  pbc={lattice.pbc}")
    n_terms = len(H.operators) if hasattr(H, "operators") else None
    print(f"Hamiltonian  : {type(H).__name__}" + (f"  ({n_terms} terms)" if n_terms else "")
          + ("  [time dependent]" if time_dependent else ""))
    if model is not None:
        print(f"Wavefunction : {type(model).__name__}({_format_fields(model)})")
    print(f"             : {wf.num_params:,} parameters")
    print(f"TDVP         : {type(tdvp).__name__}({_format_fields(tdvp)})")
    print(f"Integrator   : {integrator.name} (order {integrator.order}, "
          f"{integrator.n_stages} stages/step)")
    print(f"MC action    : {type(action).__name__}({_format_fields(action)})")
    print(f"N_mc         : {N_mc}    N_steps: {N_steps}    nsweeps: {nsweeps}"
          + (f"    start_step: {start_step}" if start_step else ""))
    print(f"time         : t0 = {t0:g}  dt = {dt:g}  ->  t_end = {t0 + N_steps * dt:g}"
          f"    ({N_steps * integrator.n_stages} sample+solve evaluations)")
    if tdvp_error_every:
        print(f"TDVP error   : every {tdvp_error_every} step(s)")
    print("-" * 60)


# ─── Driver ───────────────────────────────────────────────────────────────────

def evolve(key, H, state, wf, tdvp, action, N_steps, dt, N_mc,
           integrator="rk4", t0=0.0, wandb_run=None, log_callback_fn=None,
           nsweeps=1, opt_state=None, start_step=0, tdvp_error_every=0,
           tdvp_error_rule="rect"):
    """Run the t-VMC real-time evolution loop, printing live diagnostics.

    At every step the integrator performs ``n_stages`` evaluations of the TDVP
    right-hand side -- sample the chain with ``sample``, evaluate the local
    energies of ``H(t_stage)`` with ``compute_expectation``, solve
    ``S dtheta/dt = Im F`` with ``tdvp`` -- and combines them into the parameter
    increment. The chain is warm-started across stages and steps; see
    ``tachys.dynamics.integrators`` for why.

    Parameters
    ----------
    key             : jax.random.key
    H               : Hamiltonian operator, or a callable ``t -> operator`` for a
                      time-dependent problem. In the latter case only the
                      numerical values of the operators' ``coupling`` may vary
                      with ``t`` -- the pytree structure, static fields and leaf
                      shapes/dtypes must not, or every step would recompile
                      (checked once, up front, with an explicit error).
    state           : State — initial Monte Carlo configuration batch.
                      ``state.lattice.Ns`` sets the per-site normalizations.
    wf              : WaveFunction — must be complex-valued (see ``TDVP``).
                      Typically the output of a ground-state ``train`` run, or a
                      state prepared for a quench.
    tdvp            : TDVP instance; called as ``tdvp(E_L, opt_state, state, wf)``.
    action          : _BaseAction used for MC sampling.
    N_steps         : int — number of time steps taken by *this* call.
    dt              : float — time step. Note the cost per step is
                      ``n_stages`` sample+solve evaluations.
    N_mc            : int — number of Markov chains.
    integrator      : ``"rk4"`` (default), ``"heun"``, or an ``ExplicitRK`` instance.
    t0              : float — physical time at ``start_step``. Default 0.0.
    wandb_run       : optional wandb run. If given, logs the energy, variance,
                      acceptance, TDVP-error metrics and any callback metrics
                      every step, and checkpoints ``wf.params``/``state``/
                      ``opt_state``/``key`` exactly as ``train`` does (see
                      ``tachys.checkpoint``). Caller owns its lifecycle; expected
                      non-None only on MASTER, though every rank still
                      participates in the collective checkpoint calls.
    log_callback_fn : optional callable, or list of callables, invoked once per
                      step with either ``(state, wf, step)`` (``train``'s
                      protocol, for observables that only need the current state)
                      or ``(state, wf, step, ctx)``, where ``ctx`` is a
                      ``DynamicsContext`` carrying the time, the step's
                      Hamiltonian, the stage-1 batch, its local energies and the
                      stage-1 velocity. The arity is detected per callable.
                      Results that are not None are merged into the wandb log.
                      Called on *every* rank regardless of ``wandb_run``, since
                      callbacks are typically jitted and may touch mesh-sharded
                      arrays -- all ranks must reach the same collectives or the
                      run deadlocks.
    nsweeps         : int — MC sweeps per stage, passed to ``sample`` (default 1).
                      The main lever against the warm-start lag bias, which
                      manifests as a slow energy drift; if halving the drift
                      requires doubling ``nsweeps``, the run is lag limited.
    opt_state       : optional pre-initialized TDVP state (``TDVP`` is stateless,
                      so this only matters for checkpoint symmetry with ``train``).
    start_step      : int — absolute step number to resume at. Offsets the printed
                      step column, the wandb log step and the checkpoint
                      numbering; combine with ``t0`` to resume the physical time.
    tdvp_error_every: int — if > 0, measure the TDVP error every this many steps
                      with ``tachys.dynamics.error.TDVPError`` and show the
                      accumulated ``R^2`` in the live table. 0 (default) disables
                      it. Equivalent to passing a ``TDVPError`` yourself in
                      ``log_callback_fn``, except that this route also gets the
                      column and the ``history["R2"]`` entries.
    tdvp_error_rule : ``"rect"`` (default) or ``"trapezoid"`` — how a measurement
                      taken every ``n`` steps is extended over the steps between
                      measurements. See ``TDVPError``.

    Returns
    -------
    key, state, wf, opt_state, history
        ``history`` is a ``dict[str, list]`` with keys ``"t"``, ``"energy"``,
        ``"energy_real"``, ``"variance_per_site"``, ``"acceptance"``, and — when
        ``tdvp_error_every`` is set — ``"R2"`` and ``"tdvp_rate"``, one entry per
        step (the TDVP entries repeat the last measured value between
        measurements). The ``TDVPError`` object's own per-measurement history is
        available as ``history["tdvp_error"]``.
    """
    if not isinstance(tdvp, TDVP):
        raise TypeError(
            f"tdvp must be a tachys.dynamics.TDVP instance, got {type(tdvp).__name__}. "
            "The ground-state optimizers (SR/SPRING/MARCH) solve the imaginary-time "
            "equation and would evolve the state towards the ground state instead of "
            "propagating it in real time."
        )
    n_local = get_n_mc_local(state)
    if n_local * n_devices != N_mc:
        raise ValueError(
            f"state carries {n_local * n_devices} chains but N_mc={N_mc}; "
            "montecarlo.sample draws one key per chain, so they must agree."
        )

    integrator = get_integrator(integrator)
    H_fn, time_dependent = _as_hamiltonian_fn(H)
    H0 = _check_hamiltonian_fn(H_fn, t0, dt) if time_dependent else H

    Ns = state.Ns
    _print_setup_summary(H0, wf, tdvp, integrator, action, state, N_steps, dt, N_mc,
                         t0, nsweeps, time_dependent, tdvp_error_every, start_step)

    if callable(log_callback_fn):
        log_callback_fn = [log_callback_fn]
    callbacks = [(cb, _callback_takes_ctx(cb)) for cb in (log_callback_fn or [])]

    error_cb = TDVPError(every=tdvp_error_every, rule=tdvp_error_rule) if tdvp_error_every else None

    if wandb_run is not None:
        devices = jax.devices()
        wandb_run.config.update(
            {"num_gpus": n_devices, "gpu_kind": devices[0].device_kind if devices else None,
             "dt": dt, "integrator": integrator.name},
            allow_val_change=True,
        )

    if opt_state is None:
        opt_state = tdvp.init(wf.params)
    rhs = _TDVPRhs(H_fn, tdvp, opt_state, action, N_mc, nsweeps)

    history = {"t": [], "energy": [], "energy_real": [], "variance_per_site": [],
               "acceptance": []}
    if error_cb is not None:
        history["R2"] = []
        history["tdvp_rate"] = []
        history["tdvp_error"] = error_cb.history

    ckpt_dir, checkpoint_every, checkpoint_keep = resolve_checkpoint_settings(
        wandb_run, N_steps, rank, MASTER)
    manager = build_checkpoint_manager(ckpt_dir, checkpoint_every, checkpoint_keep) if ckpt_dir else None

    r2_col = f" {'R²':>9} │" if error_cb is not None else ""
    header = (
        f"{'step':>5} │ {'t':>9} │ {'E/N':>20} │ {'var/N':>10} │ {'dE/N':>10} │"
        f"{r2_col} {'accept':>7} │ "
        f"{'t_mc':>6} │ {'t_exp':>6} │ {'t_opt':>6} │ {'t_tot':>10} │ {'ETA (h)':>11}"
    )
    rule = "─" * len(header)

    print("\n--- START REAL-TIME EVOLUTION ---", flush=True)
    print(header, flush=True)
    print(rule, flush=True)

    e_initial = None
    t_start = time.perf_counter()
    for local_step in range(N_steps):
        step = start_step + local_step
        t = t0 + local_step * dt
        t_step0 = time.perf_counter()

        wf_before = wf
        key, wf, state, ks, auxes = integrator.step(rhs, key, t, wf, state, dt)

        # Physics is reported from stage 1: the only stage evaluated at a point
        # that lies on the trajectory. The later stages are predictor points the
        # state never visits, and their energies are off by O(dt) -- plotting them
        # produces a spurious sawtooth.
        s0 = auxes[0]
        energy, e_per_site, variance_per_site, _ = _dynamics_metrics(s0.e_mean, s0.e2_mean, Ns)
        if e_initial is None:
            e_initial = energy
        drift_per_site = (energy - e_initial) / Ns
        acc = float(jnp.mean(s0.acceptance))

        history["t"].append(t)
        history["energy"].append(e_per_site)
        history["energy_real"].append(energy)
        history["variance_per_site"].append(variance_per_site)
        history["acceptance"].append(acc)

        ctx = DynamicsContext(
            step=step, t=t, dt=dt, H=H_fn(t), wf=wf_before, state=s0.state,
            log_amps=s0.log_amps, E_L=s0.E_L, e_mean=s0.e_mean, e2_mean=s0.e2_mean,
            dtheta_dt=ks[0], acceptance=s0.acceptance, mode=tdvp.mode, Ns=Ns,
            stages=tuple(auxes),
        )

        # Callbacks run on every rank, in lockstep, even though only MASTER logs:
        # they are typically jitted and may touch mesh-sharded arrays, so a rank
        # that skipped one would leave the others blocked on a collective forever.
        callback_metrics = {}
        if error_cb is not None:
            result = error_cb(state, wf, step, ctx)
            if result is not None:
                callback_metrics.update(result)
            history["R2"].append(error_cb.R2)
            history["tdvp_rate"].append(
                error_cb.history["rate"][-1] if error_cb.history["rate"] else float("nan"))
        for cb, takes_ctx in callbacks:
            result = cb(state, wf, step, ctx) if takes_ctx else cb(state, wf, step)
            if result is not None:
                callback_metrics.update(result)

        if wandb_run is not None:
            wandb_run.log({
                "t": t,
                "dt": dt,
                "energy": e_per_site,
                "energy_drift_per_site": drift_per_site,
                "variance_per_site": variance_per_site,
                "acceptance": acc,
                **callback_metrics,
            }, step=step)

        _check_finite(e_per_site, t, wandb_run)

        if manager is not None:
            save_training_checkpoint(manager, step + 1, key, state, wf.params, opt_state,
                                     force=(local_step == N_steps - 1))

        t_mc = sum(a.t_mc for a in auxes)
        t_exp = sum(a.t_expect for a in auxes)
        t_opt = sum(a.t_solve for a in auxes)
        t_step = time.perf_counter() - t_step0
        eta_hours = t_step * (N_steps - local_step - 1) / 3600.0

        r2_val = f" {error_cb.R2:9.3e} │" if error_cb is not None else ""
        print(
            f"{step:5d} │ {t:9.4f} │ "
            f"{e_per_site:20.12f} │ {variance_per_site:10.2e} │ {drift_per_site:10.2e} │"
            f"{r2_val} {acc:7.3f} │ "
            f"{t_mc:6.2f} │ {t_exp:6.2f} │ {t_opt:6.2f} │ {t_step:6.2f} (s) │ "
            f"{eta_hours:7.2f} (h)",
            flush=True,
        )

    opt_state = rhs.opt_state
    total_time = time.perf_counter() - t_start
    t_end = t0 + N_steps * dt

    print(rule, flush=True)
    print(f"Done. {N_steps} steps ({N_steps * integrator.n_stages} stages) in "
          f"{total_time:.1f}s ({total_time / max(N_steps, 1):.3f}s/step avg), "
          f"t = {t0:g} -> {t_end:g}")
    if history["energy_real"]:
        e_drift = history["energy_real"][-1] - history["energy_real"][0]
        print(f"Energy drift over the run: {e_drift / Ns:+.3e} per site "
              f"({'conserved for time-independent H' if not time_dependent else 'H is time dependent'})")
    if error_cb is not None:
        print(f"TDVP error   : R² = {error_cb.R2:.4e}   "
              f"(last rate = {error_cb.history['rate'][-1]:.4e}, "
              f"Var(H) = {error_cb.history['var_H'][-1]:.4e})")

    if wandb_run is not None:
        wandb_run.summary["t_end"] = t_end
        if history["energy_real"]:
            wandb_run.summary["energy_drift_per_site"] = (
                history["energy_real"][-1] - history["energy_real"][0]) / Ns
        if error_cb is not None:
            wandb_run.summary["R2"] = error_cb.R2

    if manager is not None:
        manager.wait_until_finished()
        manager.close()

    return key, state, wf, opt_state, history
