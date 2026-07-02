import dataclasses
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from tachys.lattice.operator.local_estimator import compute_expectation
from tachys.montecarlo import sample


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


def _print_setup_summary(H, wf, optimizer, action, state, N_steps, lr_schedule, N_mc):
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
    print(f"N_mc         : {N_mc}    N_steps: {N_steps}")
    print(f"lr schedule  : {lr_schedule(0):.2e} -> {lr_schedule(N_steps - 1):.2e}")
    print(C.DIM + "-" * 60 + C.RESET)


def train(key, H, state, wf, optimizer, action, N_steps, lr_schedule, N_mc,
          wandb_run=None, log_callback_fn=None):
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
    log_callback_fn : optional callable(state, wf, step) -> dict | None, or list of such
                      callables — extra metrics merged into the wandb log every step.
                      Callables that return None are skipped. Ignored if wandb_run is None.

    Returns
    -------
    key, state, wf, opt_state, history
    """
    N = state.Ns
    _print_setup_summary(H, wf, optimizer, action, state, N_steps, lr_schedule, N_mc)

    if callable(log_callback_fn):
        log_callback_fn = [log_callback_fn]

    opt_state = optimizer.init(wf.params)
    history = {"energy": [], "energy_err": [], "acceptance": [], "step_time": [], "lr": []}
    best_energy = jnp.inf

    header = (
        f"{C.BOLD}{'step':>5} │ {'accept':>7} │ {'E/N':>14} │ {'err(E)/N':>10} │ {'lr':>9} │ "
        f"{'t_mc':>6} │ {'t_exp':>6} │ {'t_opt':>6} │ {'t_tot':>6} │ {'ETA(h)':>7}{C.RESET}"
    )
    rule = C.DIM + "─" * (len(header) - len(C.BOLD) - len(C.RESET)) + C.RESET

    print("\n--- SR optimization ---", flush=True)
    print(header, flush=True)
    print(rule, flush=True)

    t_start = time.perf_counter()
    for step in range(N_steps):
        key, subkey = jax.random.split(key)
        t_step0 = time.perf_counter()

        with Timer() as t_sample:
            mc_keys = jax.random.split(subkey, N_mc)
            state, log_amps, acceptance = sample(1, state, action, mc_keys, wf)
            jax.block_until_ready((state, log_amps))

        log_amps = wf.apply_fn(wf.params, state)
        with Timer() as t_expect:
            E_L, e_mean, e2_mean = compute_expectation(H, wf, state, log_amps)
            jax.block_until_ready(E_L)

        lr = lr_schedule(step)
        with Timer() as t_opt:
            updates, opt_state = optimizer(E_L, opt_state, state, wf)
            jax.block_until_ready(updates)
            wf = wf.apply_gradients(updates, lr)

        t_step = time.perf_counter() - t_step0

        # Derived quantities: variance from <|E_L|^2> - |<E_L>|^2, error bar of the mean
        energy   = float(jnp.real(e_mean))
        variance = max(float(jnp.real(e2_mean)) - float(jnp.abs(e_mean)) ** 2, 0.0)
        energy_err = (variance / N_mc) ** 0.5
        acc = float(jnp.mean(acceptance))

        history["energy"].append(energy / N)
        history["energy_err"].append(energy_err / N)
        history["acceptance"].append(acc)
        history["step_time"].append(t_step)
        history["lr"].append(lr)

        if wandb_run is not None:
            metrics = {
                "lr": lr,
                "energy": energy / N,
                "variance": variance / N ** 2,
                "acceptance": acc,
            }
            if log_callback_fn is not None:
                for cb in log_callback_fn:
                    result = cb(state, wf, step)
                    if result is not None:
                        metrics.update(result)
            wandb_run.log(metrics, step=step)

        improved = energy < best_energy
        best_energy = min(best_energy, energy)

        # ETA: last step time (no averaging) * steps remaining
        remaining_steps = N_steps - (step + 1)
        eta_hours = t_step * remaining_steps / 3600.0

        acc_color = C.GREEN if acc > 0.4 else (C.YELLOW if acc > 0.2 else C.RED)
        e_color   = C.GREEN if improved else C.RESET

        print(
            f"{step:5d} │ {acc_color}{acc:7.3f}{C.RESET} │ "
            f"{e_color}{energy / N:14.6f}{C.RESET} │ {energy_err / N:10.2e} │ {lr:9.2e} │ "
            f"{t_sample.elapsed:6.2f} │ {t_expect.elapsed:6.2f} │ {t_opt.elapsed:6.2f} │ {t_step:6.2f} (s) │ "
            f"{C.CYAN}{eta_hours:7.2f}{C.RESET} (hours)",
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

    return key, state, wf, opt_state, history
