"""Monte Carlo estimator of the fidelity between two wavefunctions.

The quantity estimated is the normalised squared overlap

    F = |<psi|phi>|^2 / (<psi|psi> <phi|phi>)     in [0, 1],

which needs no knowledge of either normalisation -- exactly what variational
wavefunctions give us, since ``wf.apply_fn`` returns ``log psi(x)`` only up to
an arbitrary additive constant.

Two independent Markov chains are run, one per wavefunction::

    x ~ |psi(x)|^2                                          (1)
    y ~ |phi(y)|^2                                          (2)
    f(x) = phi(x) / psi(x)                                  (3)
    g(y) = psi(y) / phi(y)                                  (4)
    F_loc(x, y) = f(x) g(y)                                 (5)
    F = E_x[ E_y[ F_loc(x, y) ] ]                           (6)

because ``E_x[f] = sum_x |psi(x)|^2 phi(x)/psi(x) / <psi|psi> = <psi|phi>/<psi|psi>``
and likewise ``E_y[g] = <phi|psi>/<phi|phi>``, whose product is F.

Note the ratio orientation in (3)-(4): the *denominator* is always the
wavefunction whose ``|.|^2`` generated that sample, so that the sampling
density cancels.  (Sinibaldi's note writes ``f = psi/phi`` alongside
``x ~ |psi|^2``; with that pairing ``E_x[f] = sum_x psi* psi psi/phi`` is not
the overlap.  Swapping either the ratios or the two sampling lines fixes it,
and both fixes give the same estimator -- the one implemented here.)

Control variates (Eqs. (7)-(13) of the note, enabled by default)::

    G_loc(x, y) = F_loc(x, y) - 1/2 (|F_loc(x, y)|^2 - 1)  (12)
    F = E_x[ E_y[ G_loc(x, y) ] ]                          (13)

This is unbiased, not merely low-bias, because the subtracted term has
expectation *exactly* zero:

    E_x[|f|^2] = <phi|phi>/<psi|psi>,  E_y[|g|^2] = <psi|psi>/<phi|phi>,

so ``E[|F_loc|^2] = E_x[|f|^2] E_y[|g|^2] = 1`` identically.  Near ``psi ~ phi``
both ``F_loc`` and ``|F_loc|^2`` sit close to 1 and are strongly positively
correlated, so subtracting the latter cancels much of the former's noise.
``FidelityEstimate.norm_check`` reports the sampled ``E[|F_loc|^2]``; it should
come out ~1, and a value far from 1 means the two chains have too little
overlap for any of this to be trustworthy.

That cancellation only helps while the two are genuinely close, and it *costs*
variance when they are not -- measured on random states of a 16-32 dimensional
space, variance of (13) relative to (6):

    F        0.20    0.47    0.90    0.988   0.9989   0.99988   0.999989
    gain    0.04x   0.13x   1.8x    12x     152x     1082x     9440x

so the control variate breaks even near ``F ~ 0.9`` and is overwhelmingly worth
it above that -- i.e. for the usual job of comparing two nearly-converged
wavefunctions, which is why it is the default. Below ``F ~ 0.9`` pass
``control_variate=False``; ``measure_fidelity`` reports both side by side so the
choice can be checked after the fact rather than guessed.

Usage::

    from tachys.fidelity import measure_fidelity
    key, state_psi, state_phi, res = measure_fidelity(
        key, wf_psi, wf_phi, state_psi, state_phi, action, N_mc, N_steps=100)
    print(res.fidelity, "+-", res.fidelity_err)

or, for a single already-sampled batch, ``compute_fidelity``; or, when the
log-amplitudes are already in hand, ``fidelity_from_log_amps``.
"""
import dataclasses
from functools import partial
import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from jax import shard_map
from jax.sharding import PartitionSpec as P

from tachys.ground_state_training import Timer
from tachys.lattice.state_array import get_n_mc_local
from tachys.montecarlo import sample
from tachys.parallel import mesh, n_devices, rank, MASTER
from tachys.utils import _cast_floating_to, same_treedef

PAIRINGS = ('full', 'diagonal')


class FidelityEstimate(struct.PyTreeNode):
    """Single-batch fidelity estimate and its diagnostics (all global scalars).

    Attributes
    ----------
    fidelity       : F from Eq. (13) with control variates, or Eq. (6) without --
                     real part, which is the estimator proper (F is real by
                     construction).
    fidelity_plain : F from Eq. (6), always without control variates, for
                     comparison. Equals ``fidelity`` when ``control_variate=False``.
    norm_check     : sampled ``E[|F_loc|^2]``, exactly 1 in the infinite-sample
                     limit. The note's recommended sanity check.
    imag           : imaginary part discarded from ``fidelity``. Pure Monte Carlo
                     noise; if it is not small compared to ``fidelity`` the
                     estimate is not converged.
    log_shift      : the constant ``c`` used internally to keep ``f`` and ``g``
                     in range (see ``fidelity_from_log_amps``). Diagnostic only;
                     it cancels exactly out of every other field.
    """
    fidelity: jnp.ndarray
    fidelity_plain: jnp.ndarray
    norm_check: jnp.ndarray
    imag: jnp.ndarray
    log_shift: jnp.ndarray


def _global_mean(x):
    """Mean of ``x`` over the MC axis of *all* devices, from inside a shard_map.

    Mirrors ``local_estimator.compute_expectation``: local mean, psum, divide by
    the device count (valid because every shard holds the same number of chains).
    """
    return jax.lax.psum(jnp.mean(x), 'i') / n_devices


@jax.jit
def log_amplitudes(wf, state):
    """``log psi(x)`` for every walker of a sharded ``state``.

    The batch axis of ``state`` is sharded across the mesh and the result is
    sharded the same way. Floating-point leaves of both ``wf`` and ``state`` are
    cast to ``wf.dtype`` first, exactly as ``sample`` and ``compute_expectation``
    do -- ansatz code that derives its working precision from the incoming
    arrays would otherwise mix f32 and f64 operands.

    This is the one place where a wavefunction is evaluated on configurations it
    did not itself generate, which is the whole content of the fidelity estimator.
    """
    def _body(wf, state):
        wf, state = _cast_floating_to((wf, state), wf.dtype)
        return wf.apply_fn(wf.params, state).astype(jnp.complex128)

    return shard_map(
        _body, mesh=mesh, in_specs=(P(), P('i')), out_specs=P('i'), check_vma=False,
    )(wf, state)


@partial(jax.jit, static_argnames=('control_variate', 'pairing'))
def fidelity_from_log_amps(log_psi_x, log_phi_x, log_psi_y, log_phi_y,
                            control_variate=True, pairing='full'):
    """Fidelity from the four log-amplitude arrays, without touching a network.

    This is the numerical core: everything above it only exists to produce these
    four arrays. Keeping it free of ``WaveFunction``/``State`` makes it trivially
    testable against exact amplitudes and reusable for any sampler.

    Parameters
    ----------
    log_psi_x, log_phi_x : jax.Array, shape (N_x,), sharded on 'i'
        ``log psi`` and ``log phi`` evaluated on the configurations ``x`` sampled
        from ``|psi|^2``.
    log_psi_y, log_phi_y : jax.Array, shape (N_y,), sharded on 'i'
        ``log psi`` and ``log phi`` evaluated on the configurations ``y`` sampled
        from ``|phi|^2``.
    control_variate : bool
        Use Eq. (12)-(13) instead of Eq. (6). Recommended (and default): it is
        unbiased and much lower variance.
    pairing : {'full', 'diagonal'}
        How the outer expectation ``E_x[E_y[.]]`` over the two independent chains
        is formed.

        - ``'full'`` (default) averages over all ``N_x * N_y`` cross pairs. Both
          ``F_loc`` and ``|F_loc|^2`` factorise into ``(.)_x * (.)_y``, so this
          costs the same O(N) as the diagonal and just averages each factor
          separately -- strictly more samples, strictly less variance, and
          ``N_x`` need not equal ``N_y``.
        - ``'diagonal'`` pairs walker ``i`` of ``x`` with walker ``i`` of ``y``
          and averages the resulting ``N`` values of ``G_loc``. This is the
          literal reading of Eqs. (5)-(6); it is retained because it yields a
          genuine per-walker estimator, which is what a jackknife or a
          walker-resolved diagnostic would need. Requires ``N_x == N_y``.

    Returns
    -------
    FidelityEstimate
    """
    if pairing not in PAIRINGS:
        raise ValueError(f"pairing must be one of {PAIRINGS}, got {pairing!r}")
    if pairing == 'diagonal' and log_psi_x.shape != log_psi_y.shape:
        raise ValueError(
            f"pairing='diagonal' pairs the two chains walker by walker and needs equal "
            f"chain counts, got N_x={log_psi_x.shape[0]} and N_y={log_psi_y.shape[0]}. "
            f"Use pairing='full' for unequal chains."
        )

    def _body(log_psi_x, log_phi_x, log_psi_y, log_phi_y):
        log_f = log_phi_x - log_psi_x      # Eq. (3)
        log_g = log_psi_y - log_phi_y      # Eq. (4)

        # psi and phi each carry an arbitrary additive constant in their log
        # amplitude, and the two are unrelated, so log_f and log_g can be offset
        # by a large constant +-c that cancels exactly in every product f*g below.
        # Exponentiating them as they come would then overflow one factor and
        # flush the other to zero. Shifting by c = (<Re log_f> - <Re log_g>)/2
        # re-centres both on the same value without changing any result.
        shift = 0.5 * (_global_mean(jnp.real(log_f)) - _global_mean(jnp.real(log_g)))
        f = jnp.exp(log_f - shift)
        g = jnp.exp(log_g + shift)

        if pairing == 'full':
            # <F_loc> = <f>_x <g>_y and <|F_loc|^2> = <|f|^2>_x <|g|^2>_y, so all
            # N_x*N_y cross pairs are covered by four O(N) averages.
            F_plain    = _global_mean(f) * _global_mean(g)
            norm_check = _global_mean(jnp.abs(f) ** 2) * _global_mean(jnp.abs(g) ** 2)
        else:
            F_loc      = f * g                                    # Eq. (5)/(11)
            F_plain    = _global_mean(F_loc)                      # Eq. (6)
            norm_check = _global_mean(jnp.abs(F_loc) ** 2)

        # Eq. (12)-(13). Linear in F_loc and |F_loc|^2, so applying it to the
        # means is identical to averaging G_loc walker by walker.
        F = F_plain - 0.5 * (norm_check - 1.0) if control_variate else F_plain

        return FidelityEstimate(
            fidelity=jnp.real(F),
            fidelity_plain=jnp.real(F_plain),
            norm_check=jnp.real(norm_check),
            imag=jnp.imag(F),
            log_shift=shift,
        )

    return shard_map(
        _body, mesh=mesh,
        in_specs=(P('i'), P('i'), P('i'), P('i')),
        out_specs=P(),
        check_vma=False,
    )(log_psi_x, log_phi_x, log_psi_y, log_phi_y)


@partial(jax.jit, static_argnames=('control_variate', 'pairing'))
def compute_fidelity(wf_psi, wf_phi, state_x, state_y,
                     log_psi_x=None, log_phi_y=None,
                     control_variate=True, pairing='full'):
    """Fidelity between ``wf_psi`` and ``wf_phi`` from one pair of sampled batches.

    Parameters
    ----------
    wf_psi, wf_phi : WaveFunction
        The two wavefunctions. They may be entirely different ansaetze (different
        ``apply_fn``, params and ``dtype``); they only have to accept the same
        ``State`` type, since each is evaluated on the other's configurations.
    state_x : State
        Configurations distributed as ``|psi|^2``, batch axis sharded on 'i'.
    state_y : State
        Configurations distributed as ``|phi|^2``, batch axis sharded on 'i'.
    log_psi_x : jax.Array, optional
        ``log psi`` on ``state_x``, if already known -- ``montecarlo.sample``
        returns exactly this for the chain it propagated, so passing it through
        saves one full network evaluation per call.
    log_phi_y : jax.Array, optional
        Likewise ``log phi`` on ``state_y``.
    control_variate, pairing
        See ``fidelity_from_log_amps``.

    Returns
    -------
    FidelityEstimate

    Notes
    -----
    The two cross evaluations (``phi`` on ``x`` and ``psi`` on ``y``) are always
    computed here and are the dominant cost: two forward passes per call.
    """
    if not same_treedef(state_x, state_y):
        raise TypeError(
            "state_x and state_y must be the same State type over the same Hilbert "
            "space -- each wavefunction is evaluated on the other's configurations. "
            f"Got {type(state_x).__name__} and {type(state_y).__name__}."
        )

    log_phi_x = log_amplitudes(wf_phi, state_x)
    log_psi_y = log_amplitudes(wf_psi, state_y)
    if log_psi_x is None:
        log_psi_x = log_amplitudes(wf_psi, state_x)
    if log_phi_y is None:
        log_phi_y = log_amplitudes(wf_phi, state_y)

    return fidelity_from_log_amps(log_psi_x, log_phi_x, log_psi_y, log_phi_y,
                                   control_variate=control_variate, pairing=pairing)


def blocking_error(x, min_blocks=8):
    """Error bar on the mean of a correlated series, by binning analysis.

    Successively halves the series by averaging adjacent pairs and records the
    naive standard error at each binning level. Under correlation the naive error
    grows with the bin size until the bins are longer than the autocorrelation
    time, then plateaus; the maximum over levels is a standard conservative proxy
    for that plateau. Falls back to the naive error for series too short to bin.

    Parameters
    ----------
    x          : array-like, shape (n,) -- one measurement per MC step.
    min_blocks : stop binning once fewer than this many bins remain.
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size < 2:
        return float('nan')
    errs, y = [], x
    while y.size >= max(2, min_blocks):
        errs.append(y.std(ddof=1) / np.sqrt(y.size))
        if y.size % 2:
            y = y[:-1]
        y = y.reshape(-1, 2).mean(axis=1)
    if not errs:
        errs.append(x.std(ddof=1) / np.sqrt(x.size))
    return float(np.max(errs))


@dataclasses.dataclass
class FidelityResult:
    """Outcome of ``measure_fidelity``: scalars, plus the full per-step history.

    ``fidelity_err`` and ``fidelity_plain_err`` come from ``blocking_error`` over
    the per-step series, so they account for the autocorrelation between
    consecutive MC steps rather than assuming independent measurements.
    """
    fidelity: float
    fidelity_err: float
    fidelity_plain: float
    fidelity_plain_err: float
    norm_check: float
    imag: float
    history: dict
    acceptance: tuple


def measure_fidelity(key, wf_psi, wf_phi, state_psi, state_phi, action, N_mc, N_steps,
                     action_phi=None, nsweeps=1, warmup=0, control_variate=True,
                     pairing='full', log_every=1):
    """Sample both wavefunctions and measure their fidelity over ``N_steps`` steps.

    Two *independent* Markov chains are advanced in lockstep, one under
    ``|psi|^2`` and one under ``|phi|^2`` (Eqs. (1)-(2)); their independence is
    what makes ``E[|F_loc|^2] = 1`` hold exactly, so they must never be seeded
    from a shared key. Neither wavefunction is modified.

    Parameters
    ----------
    key             : jax.random.key
    wf_psi, wf_phi  : WaveFunction -- may be different ansaetze, see ``compute_fidelity``.
    state_psi       : State -- starting configurations for the ``|psi|^2`` chain.
    state_phi       : State -- starting configurations for the ``|phi|^2`` chain.
                      Pass a genuinely separate object; reusing one checkpoint's
                      state for both chains only costs you the ``warmup`` needed
                      to decorrelate them.
    action          : _BaseAction -- MC move used for the ``psi`` chain, and for
                      the ``phi`` chain too unless ``action_phi`` is given.
    N_mc            : int -- number of chains, per wavefunction.
    N_steps         : int -- number of measurement steps.
    action_phi      : _BaseAction, optional -- separate move set for the ``phi``
                      chain (e.g. a different ansatz that needs a different
                      proposal). Defaults to ``action``.
    nsweeps         : int -- MC sweeps per step, passed to ``montecarlo.sample``.
    warmup          : int -- equilibration steps run before measuring, discarded.
                      Worth setting whenever either chain starts from
                      configurations that were not equilibrated under *its own*
                      wavefunction -- which is the normal case for at least one
                      of the two when both start from the same checkpoint.
    control_variate, pairing : see ``fidelity_from_log_amps``.
    log_every       : print a status line every this many steps (0 disables).

    Returns
    -------
    key, state_psi, state_phi, result
        ``result`` is a ``FidelityResult``; ``result.history`` holds the per-step
        arrays ``fidelity``, ``fidelity_plain``, ``norm_check`` and ``imag``.
    """
    if pairing not in PAIRINGS:
        raise ValueError(f"pairing must be one of {PAIRINGS}, got {pairing!r}")
    if action_phi is None:
        action_phi = action
    for name, st in (("state_psi", state_psi), ("state_phi", state_phi)):
        n = get_n_mc_local(st)
        if n != N_mc:
            raise ValueError(
                f"{name} carries {n} chains but N_mc={N_mc}; montecarlo.sample draws one key "
                f"per chain, so they must agree. Rebuild the state with N_mc={N_mc} walkers."
            )

    print("\n--- Fidelity measurement ---")
    print(f"N_mc: {N_mc}    N_steps: {N_steps}    warmup: {warmup}    nsweeps: {nsweeps}")
    print(f"control_variate: {control_variate}    pairing: {pairing}")
    print("-" * 60)

    history = {k: [] for k in ("fidelity", "fidelity_plain", "norm_check", "imag")}
    acc_psi_hist, acc_phi_hist = [], []

    header = (f"{'step':>5} │ {'F':>12} │ {'F (no CV)':>12} │ {'E|F_loc|^2':>11} │ "
              f"{'acc_psi':>7} │ {'acc_phi':>7} │ {'t_mc':>6} │ {'t_F':>6} │ {'ETA (h)':>8}")
    rule = "─" * len(header)
    if log_every:
        print(header, flush=True)
        print(rule, flush=True)

    for step in range(-warmup, N_steps):
        t_step0 = time.perf_counter()
        # Independent keys per chain: the factorisation E[|F_loc|^2] = E|f|^2 E|g|^2
        # that makes the control variate exact assumes x and y are uncorrelated.
        key, key_psi, key_phi = jax.random.split(key, 3)

        with Timer() as t_sample:
            state_psi, log_psi_x, acc_psi = sample(
                nsweeps, state_psi, action, jax.random.split(key_psi, N_mc), wf_psi)
            state_phi, log_phi_y, acc_phi = sample(
                nsweeps, state_phi, action_phi, jax.random.split(key_phi, N_mc), wf_phi)
            jax.block_until_ready((state_psi, state_phi))

        if step < 0:                      # warmup: equilibrate only, measure nothing
            if log_every and step % log_every == 0:
                print(f"{step:5d} │ {'(warmup)':>12} │ {'':>12} │ {'':>11} │ "
                      f"{float(jnp.mean(acc_psi)):7.3f} │ {float(jnp.mean(acc_phi)):7.3f} │ "
                      f"{t_sample.elapsed:6.2f} │ {'':>6} │", flush=True)
            continue

        with Timer() as t_fid:
            est = compute_fidelity(wf_psi, wf_phi, state_psi, state_phi,
                                    log_psi_x=log_psi_x, log_phi_y=log_phi_y,
                                    control_variate=control_variate, pairing=pairing)
            jax.block_until_ready(est)

        history["fidelity"].append(float(est.fidelity))
        history["fidelity_plain"].append(float(est.fidelity_plain))
        history["norm_check"].append(float(est.norm_check))
        history["imag"].append(float(est.imag))
        acc_psi_hist.append(float(jnp.mean(acc_psi)))
        acc_phi_hist.append(float(jnp.mean(acc_phi)))

        t_step = time.perf_counter() - t_step0
        if log_every and rank == MASTER and (step % log_every == 0 or step == N_steps - 1):
            print(f"{step:5d} │ {history['fidelity'][-1]:12.8f} │ "
                  f"{history['fidelity_plain'][-1]:12.8f} │ {history['norm_check'][-1]:11.6f} │ "
                  f"{acc_psi_hist[-1]:7.3f} │ {acc_phi_hist[-1]:7.3f} │ "
                  f"{t_sample.elapsed:6.2f} │ {t_fid.elapsed:6.2f} │ "
                  f"{t_step * (N_steps - step - 1) / 3600.0:8.3f}", flush=True)

    history = {k: np.asarray(v) for k, v in history.items()}
    result = FidelityResult(
        fidelity=float(history["fidelity"].mean()),
        fidelity_err=blocking_error(history["fidelity"]),
        fidelity_plain=float(history["fidelity_plain"].mean()),
        fidelity_plain_err=blocking_error(history["fidelity_plain"]),
        norm_check=float(history["norm_check"].mean()),
        imag=float(history["imag"].mean()),
        history=history,
        acceptance=(float(np.mean(acc_psi_hist)), float(np.mean(acc_phi_hist))),
    )

    if log_every:
        print(rule, flush=True)
    print(f"F              = {result.fidelity:.8f} ± {result.fidelity_err:.2e}"
          f"   (1 - F = {1.0 - result.fidelity:.3e})")
    print(f"F (no CV)      = {result.fidelity_plain:.8f} ± {result.fidelity_plain_err:.2e}")
    print(f"E[|F_loc|^2]   = {result.norm_check:.6f}   (must be ~1)")
    print(f"Im F           = {result.imag:.3e}   (pure noise; small vs F)")
    if not np.isfinite(result.norm_check) or abs(result.norm_check - 1.0) > 0.1:
        print("WARNING: E[|F_loc|^2] is far from 1 -- psi and phi overlap too little for "
              "this estimator to be reliable. Treat F as a lower bound at best.", flush=True)

    return key, state_psi, state_phi, result
