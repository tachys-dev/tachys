import dataclasses
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from tachys.checkpoint import build_checkpoint_manager, resolve_checkpoint_settings, save_training_checkpoint
from tachys.lattice.operator.local_estimator import compute_expectation
from tachys.montecarlo import sample
from tachys.parallel import rank, MASTER


class Timer:
    def __enter__(self):
        self._t0 = time.perf_counter()
        return self
    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._t0


_USE_COLOR = sys.stdout.isatty()


class C:
    RESET  = "\033[0m"  if _USE_COLOR else ""
    BOLD   = "\033[1m"  if _USE_COLOR else ""
    DIM    = "\033[2m"  if _USE_COLOR else ""
    GREEN  = "\033[32m" if _USE_COLOR else ""
    YELLOW = "\033[33m" if _USE_COLOR else ""
    RED    = "\033[31m" if _USE_COLOR else ""
    CYAN   = "\033[36m" if _USE_COLOR else ""


def _format_fields(obj):
    """Best-effort 'field=value, ...' string for a dataclass / flax-struct instance."""
    try:
        fields = dataclasses.fields(obj)
    except TypeError:
        return ""
    parts = []
    for f in fields:
        val = getattr(obj, f.name, None)
        if isinstance(val, (jnp.ndarray, np.ndarray)) or callable(val):
            continue
        parts.append(f"{f.name}={val}")
    return ", ".join(parts)


def _compute_metrics(mean_e, mean_E2, Ns):
    """Derive scalar metrics from energy moments."""
    e = jnp.real(mean_e).item()
    e2 = jnp.real(mean_E2).item()
    vscore = Ns * (e2 - e**2) / e**2
    variance_per_site = (e2 - e**2) / Ns
    e_per_site = e / Ns
    return e, e2, e_per_site, vscore, variance_per_site


def _print_setup_summary(H, wf, optimizer, action, state, N_steps, lr_schedule, N_mc, start_step=0):
    lattice = state.lattice
    model = getattr(wf.apply_fn, "__self__", None)

    print(f"\n{C.BOLD}--- Simulation setup ---{C.RESET}")
    if lattice is not None:
        print(f"Lattice      : {type(lattice).__name__}  Ns={lattice.Ns}  L={lattice.L}  "
              f"nb={lattice.nb}  pbc={lattice.pbc}")
    n_terms = len(H.operators) if hasattr(H, "operators") else None
    print(f"Hamiltonian  : {type(H).__name__}" + (f"  ({n_terms} terms)" if n_terms else ""))
    if model is not None:
        print(f"Wavefunction : {type(model).__name__}({_format_fields(model)})")
    print(f"             : {wf.num_params:,} parameters")
    print(f"Optimizer    : {type(optimizer).__name__}({_format_fields(optimizer)})")
    print(f"MC action    : {type(action).__name__}({_format_fields(action)})")
    print(f"N_mc         : {N_mc}    N_steps: {N_steps}" + (f"    start_step: {start_step}" if start_step else ""))
    print(f"lr schedule  : {lr_schedule(start_step):.2e} -> {lr_schedule(start_step + N_steps - 1):.2e}")
    print(C.DIM + "-" * 60 + C.RESET)


def train(key, H, state, wf, optimizer, action, N_steps, lr_schedule, N_mc,
          wandb_run=None, log_callback_fn=None, skip_optimization=False, nsweeps=1,
          opt_state=None, start_step=0):
    """Run the SR optimization loop, printing live diagnostics.

    Parameters
    ----------
    key             : jax.random.key
    H               : Hamiltonian operator
    state           : State — its lattice determines N, the number of sites used to report energy per site
    wf              : WaveFunction
    optimizer       : optimizer (e.g. SR, SPRING, MARCH)
    action          : _BaseAction used for MC sampling
    N_steps         : int — number of optimization steps
    lr_schedule     : callable(step: int) -> float — learning rate as a function of step
    N_mc            : int — number of Markov chains
    wandb_run       : optional wandb run (e.g. from ``wandb.init(...)``) — if given, logs
                      lr, energy, variance and acceptance every step. Caller owns its
                      lifecycle (init/finish); tachys.training never imports wandb itself.
                      Also checkpoints wf.params, state, opt_state and key (via
                      orbax, see tachys.checkpoint) into wandb_run.dir/checkpoints every
                      ``wandb_run.config["checkpoint_every"]`` steps (defaults to N_steps,
                      i.e. once at the end) and on the final step. wandb_run is expected
                      to be non-None only on the MASTER rank (as in existing callers);
                      every rank still participates in the collective checkpoint calls.
    log_callback_fn : optional callable(state, wf, step) -> dict | None, or list of such
                      callables — extra metrics merged into the wandb log every step.
                      Callables that return None are skipped. Called on every rank
                      regardless of wandb_run (callbacks are typically jitted and may
                      touch mesh-sharded arrays, so all ranks must call them in lockstep);
                      only the merged result is actually logged, and only on MASTER.
    skip_optimization : bool — if True, skip the optimizer step and parameter update each
                      iteration, only sampling and evaluating the energy of ``wf``.
    nsweeps         : int — number of MC sweeps per step passed to ``sample`` (default 1).
    opt_state       : optional pre-initialized optimizer state (e.g. restored from a
                      checkpoint via tachys.checkpoint.load_checkpoint) to resume training
                      from. Defaults to a fresh ``optimizer.init(wf.params)``.
    start_step      : int — absolute step number to resume at (e.g. the step count
                      recovered alongside ``opt_state`` when resuming from a checkpoint).
                      Offsets ``lr_schedule``, the wandb log step, the printed step
                      column and the checkpoint step numbering so they continue from
                      where the previous run left off instead of restarting at 0.
                      ``N_steps`` still counts iterations run by *this* call — pass the
                      remaining steps, not the original total.

    Returns
    -------
    key, state, wf, opt_state, history
    """
    N = state.Ns
    _print_setup_summary(H, wf, optimizer, action, state, N_steps, lr_schedule, N_mc, start_step)

    if callable(log_callback_fn):
        log_callback_fn = [log_callback_fn]

    if opt_state is None:
        opt_state = optimizer.init(wf.params)
    history = {"energy": [], "variance_per_site": [], "vscore": [], "acceptance": [], "lr": []}
    best_energy = jnp.inf

    ckpt_dir, checkpoint_every, checkpoint_keep = resolve_checkpoint_settings(wandb_run, N_steps, rank, MASTER)
    manager = build_checkpoint_manager(ckpt_dir, checkpoint_every, checkpoint_keep) if ckpt_dir else None

    header = (
        f"{C.BOLD}{'step':>5} │ {'E/N':>20} │ {'var/N':>10} │ {'vscore':>8} │ {'accept':>7} │ {'lr':>9} │ "
        f"{'t_mc':>6} │ {'t_exp':>6} │ {'t_opt':>6} │ {'t_tot':>10} │ {'ETA (h)':>11}{C.RESET}"
    )
    rule = C.DIM + "─" * (len(header) - len(C.BOLD) - len(C.RESET)) + C.RESET

    print("\n--- START TRAINING ---", flush=True)
    print(header, flush=True)
    print(rule, flush=True)

    t_start = time.perf_counter()
    for local_step in range(N_steps):
        step = start_step + local_step
        key, subkey = jax.random.split(key)
        t_step0 = time.perf_counter()

        with Timer() as t_sample:
            mc_keys = jax.random.split(subkey, N_mc)
            state, log_amps, acceptance = sample(nsweeps, state, action, mc_keys, wf)
            jax.block_until_ready((state, log_amps))

        with Timer() as t_expect:
            E_L, e_mean, e2_mean = compute_expectation(H, wf, state, log_amps)
            jax.block_until_ready(E_L)

        lr = lr_schedule(step)
        with Timer() as t_opt:
            if not skip_optimization:
                updates, opt_state = optimizer(E_L, opt_state, state, wf)
                jax.block_until_ready(updates)
                wf = wf.apply_gradients(updates, lr)

        # Derived quantities: variance from <|E_L|^2> - |<E_L>|^2
        energy, _, e_per_site, vscore, variance_per_site = _compute_metrics(e_mean, e2_mean, N)
        acc = float(jnp.mean(acceptance))

        history["energy"].append(e_per_site)
        history["variance_per_site"].append(variance_per_site)
        history["vscore"].append(vscore)
        history["acceptance"].append(acc)
        history["lr"].append(lr)

        # log_callback_fn entries are typically jax.jit-compiled and may touch
        # arrays sharded across the global (multi-process) device mesh, so
        # every rank must call them in lockstep even though only MASTER
        # actually logs the result — otherwise MASTER blocks on a collective
        # the other ranks never join, deadlocking the whole run.
        callback_metrics = {}
        if log_callback_fn is not None:
            for cb in log_callback_fn:
                result = cb(state, wf, step)
                if result is not None:
                    callback_metrics.update(result)

        if wandb_run is not None:
            metrics = {
                "lr": lr,
                "energy": e_per_site,
                "variance_per_site": variance_per_site,
                "vscore": vscore,
                "acceptance": acc,
                **callback_metrics,
            }
            wandb_run.log(metrics, step=step)

        if manager is not None:
            save_training_checkpoint(manager, step + 1, key, state, wf.params, opt_state,
                                      force=(local_step == N_steps - 1))

        improved = energy < best_energy
        best_energy = min(best_energy, energy)

        # ETA: last step time (no averaging) * steps remaining
        t_step = time.perf_counter() - t_step0
        remaining_steps = N_steps - (local_step + 1)
        eta_hours = t_step * remaining_steps / 3600.0

        acc_color = C.GREEN if acc > 0.4 else (C.YELLOW if acc > 0.2 else C.RED)
        e_color   = C.GREEN if improved else C.RESET

        print(
            f"{step:5d} │ "
            f"{e_color}{e_per_site:20.12f}{C.RESET} │ {variance_per_site:10.2e} │ {vscore:8.4f} │ "
            f"{acc_color}{acc:7.3f}{C.RESET} │ {lr:9.2e} │ "
            f"{t_sample.elapsed:6.2f} │ {t_expect.elapsed:6.2f} │ {t_opt.elapsed:6.2f} │ {t_step:6.2f} (s) │ "
            f"{C.CYAN}{eta_hours:7.2f}{C.RESET} (h)",
            flush=True,
        )

    total_time = time.perf_counter() - t_start
    window = min(100, len(history["energy"]))
    tail_energy = history["energy"][-window:]
    mean_energy_tail = np.mean(tail_energy)
    var_energy_tail  = np.var(tail_energy)

    print(rule, flush=True)
    print(
        f"{C.BOLD}Done.{C.RESET} {N_steps} steps in {total_time:.1f}s "
        f"({total_time / N_steps:.3f}s/step avg)"
    )
    print(f"E/N over last {window} steps = {mean_energy_tail:.6f} ± {var_energy_tail ** 0.5:.2e}  (var = {var_energy_tail:.2e})")

    if wandb_run is not None:
        wandb_run.summary["mean_energy"] = mean_energy_tail
        wandb_run.summary["var_energy"] = var_energy_tail

    if manager is not None:
        manager.wait_until_finished()
        manager.close()

    return key, state, wf, opt_state, history


def compute_observables(key, N_steps, state, action, wf, N_mc, op_groups, nsweeps=1, log_every=1):
    """Measure a fixed set of observables along a Markov chain.

    Unlike ``train``, ``wf`` is held fixed here — this only samples and
    evaluates ``op_groups``, it never updates parameters. Every operator is
    evaluated on the same sampled batch at each step, so different
    observables share Monte Carlo statistics rather than being measured from
    independent runs.

    Parameters
    ----------
    key       : jax.random.key
    N_steps   : int — number of sampling steps (measurements)
    state     : State — current Monte Carlo configuration
    action    : _BaseAction used for MC sampling
    wf        : WaveFunction — fixed guiding wavefunction
    N_mc      : int — number of Markov chains
    op_groups : dict[str, Sequence[_Operator]] — named groups of observables
                (e.g. the output of an observable-construction helper like
                ``spin_spin_alm``), every operator of every group evaluated at
                every step. A bare sequence of operators is also accepted and
                treated as a single group named ``"obs"``.
    nsweeps   : int — number of MC sweeps per step passed to ``sample`` (default 1).
    log_every : print a status line every this many steps (0 disables).

    Returns
    -------
    key, state, metrics
        metrics : dict[str, np.ndarray] — ``metrics[name]`` has shape
                  ``(N_steps, len(op_groups[name]))``, the real part of ``<O>``
                  at every step.
    """
    if not isinstance(op_groups, dict):
        op_groups = {"obs": op_groups}

    n_ops = sum(len(ops) for ops in op_groups.values())
    print(f"\n{C.BOLD}--- Observable measurement ---{C.RESET}")
    print(f"N_mc: {N_mc}    N_steps: {N_steps}    nsweeps: {nsweeps}    n_observables: {n_ops}")
    print(C.DIM + "-" * 60 + C.RESET)

    metrics = {name: [] for name in op_groups}

    for step in range(N_steps):
        key, subkey = jax.random.split(key)
        t_step0 = time.perf_counter()

        with Timer() as t_sample:
            mc_keys = jax.random.split(subkey, N_mc)
            state, log_amps, acceptance = sample(nsweeps, state, action, mc_keys, wf)
            jax.block_until_ready((state, log_amps))

        with Timer() as t_expect:
            step_means = {}
            op_i = 0
            for name, ops in op_groups.items():
                means = []
                for op in ops:
                    if _USE_COLOR and rank == MASTER:
                        print(f"\r{C.DIM}  step {step:4d} - observable {op_i+1:>4d}/{n_ops} ({name}){C.RESET}",
                              end="", flush=True)
                    means.append(compute_expectation(op, wf, state, log_amps)[1])
                    op_i += 1
                step_means[name] = jnp.stack(means)
            jax.block_until_ready(step_means)
            if _USE_COLOR and rank == MASTER:
                print("\r" + " " * 60 + "\r", end="", flush=True)

        for name, means in step_means.items():
            metrics[name].append(np.asarray(means.real))

        t_step = time.perf_counter() - t_step0
        eta_hours = t_step * (N_steps - step - 1) / 3600.0

        if log_every and rank == MASTER and (step % log_every == 0 or step == N_steps - 1):
            print(
                f"{step:5d} │ accept: {float(jnp.mean(acceptance)):.3f} │ "
                f"t_mc: {t_sample.elapsed:6.2f}s │ t_obs: {t_expect.elapsed:6.2f}s │ "
                f"t_tot: {t_step:6.2f}s │ ETA: {eta_hours:7.2f}h",
                flush=True,
            )

    for name in metrics:
        metrics[name] = np.array(metrics[name])

    return key, state, metrics
