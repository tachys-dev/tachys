"""Single Lanczos step (LS) on top of a variational wavefunction.

Given a guiding wavefunction ``psi_G`` and a Hamiltonian ``H``, the one-step
Lanczos (Krylov) state is

    |psi_alpha> = (1 + alpha H) |psi_G>,                                  (LS)

whose amplitudes are available configuration by configuration because

    psi_alpha(x) = <x|(1 + alpha H)|psi_G> = (1 + alpha e_G(x)) psi_G(x),
    e_G(x)       = <x|H|psi_G> / <x|psi_G>                                (1)

is the ordinary local energy of ``psi_G`` -- exactly what ``local_estimator``
returns.  (Appendix C writes the local energy as ``<psi|H|x>/<psi|x>``, which
is the complex conjugate of (1); for a complex ``psi_G`` it is (1), not its
conjugate, that gives the amplitudes of ``(1 + alpha H)|psi_G>``.)
``psi_alpha`` is therefore a perfectly usable ``WaveFunction``: it can be
sampled, its energy and its correlation functions measured, and -- Sorella's
point -- its energy is variational and strictly below that of ``psi_G`` at the
optimal ``alpha``.

``alpha`` is real: the Krylov space of a Hermitian ``H`` is spanned with real
coefficients.

Reference
---------
S. Sorella, *Generalized Lanczos algorithm for variational quantum Monte
Carlo*, cond-mat/0009149 (Phys. Rev. B **64**, 024512), Appendix C.  Equation
numbers below are that paper's.

The algorithm (Appendix C)
--------------------------
The energy of the LS state is a ratio of the first three energy moments
``h_n = <psi_G|H^n|psi_G> / <psi_G|psi_G>``,

    E(alpha) = (h1 + 2 alpha h2 + alpha^2 h3)
             / (1  + 2 alpha h1 + alpha^2 h2),                          (C3)

and is minimised analytically at

    alpha* = [-(h3 - h1 h2) +- sqrt((h3 - h1 h2)^2
              - 4 (h2 - h1^2)(h1 h3 - h2^2))] / [2 (h1 h3 - h2^2)],     (C4)

the sign being the one with the lower ``E(alpha*)``.  Sampling ``h2`` and
especially ``h3`` directly from ``|psi_G|^2`` (Eq. C2) is statistically
hopeless.  Appendix C instead recovers them from two cheap averages taken
over ``|psi_alpha|^2`` at an arbitrary probe ``alpha``:

    E(alpha) = <psi_alpha|H|psi_alpha> / <psi_alpha|psi_alpha>
             = E_{|psi_alpha|^2}[ e_alpha(x) ]                            (2)
    chi      = <psi_alpha|(1 + alpha H)^-1|psi_alpha> / <psi_alpha|psi_alpha>
             = E_{|psi_alpha|^2}[ (1 + alpha e_G(x))^-1 ]                 (3)

together with ``h1 = <H>_{psi_G}`` from a standard run on ``|psi_G|^2``.
Then

    h2 = [(1/chi - 2)(1 + alpha h1) + 1] / alpha^2                        (4)
    h3 = [E(alpha)(1 + 2 alpha h1 + alpha^2 h2) - h1 - 2 alpha h2]
         / alpha^2                                                        (5)

which are (4)-(5) of Appendix C, and are exact inversions of ``chi`` and
``E(alpha)`` written out in terms of the moments -- no approximation:

    chi(alpha) = (1 + alpha h1) / (1 + 2 alpha h1 + alpha^2 h2).          (6)

Why (3) is an average of a *local* quantity, despite ``(1 + alpha H)^-1``
being dense: ``(1 + alpha H)^-1 |psi_alpha> = |psi_G>`` by construction, so

    <psi_alpha|(1 + alpha H)^-1|psi_alpha> = <psi_alpha|psi_G>
        = sum_x |psi_alpha(x)|^2  psi_G(x) / psi_alpha(x),

and ``psi_G/psi_alpha = 1/(1 + alpha e_G)`` by (1).  This module evaluates
``chi`` in the ``psi_G/psi_alpha`` form: it costs one forward pass of
``psi_G`` on the sampled configurations, where ``1/(1 + alpha e_G)`` would
cost a whole local-energy evaluation.  The two are the same number term by
term, not merely in expectation.

The gain over brute force is that the *hardest* moment, ``h3``, is obtained
from an energy expectation value, which is a far better-behaved estimator
than ``<H^3>``; and ``h2``, hence the guiding state's variance, comes out of
``chi`` without ever applying ``H`` twice to ``psi_G``.

The sign and scale of alpha*
----------------------------
``dE/dalpha|_0 = 2 (h2 - h1^2) = 2 Var_G[H] > 0``, so ``alpha = 0`` is never
optimal and ``E(alpha)`` always rises on the positive side -- but that fixes
only the slope at the origin, *not* the sign of the minimiser.  The
stationarity quadratic behind (C4) is

    a alpha^2 + b alpha + c = 0,
    a = h1 h3 - h2^2,  b = h3 - h1 h2,  c = h2 - h1^2 = Var_G > 0,        (7)

so the two roots have product ``c/a``: they share the sign of ``a`` when
``a > 0`` and straddle zero when ``a < 0``.  The sign of ``alpha*`` is
therefore state dependent.  What it means is clearest for a guiding state
dominated by ``E_0`` with a single contaminating level ``E_1``, where (7)
factorises exactly into

    alpha = -1/E_1  (the minimum: the filter 1 + alpha H annihilates the
                     contaminant, and E(alpha*) = E_0 exactly)
    alpha = -1/E_0  (the maximum, where E(alpha) = E_1).

Hence ``alpha* ~ -1/E_1``, of magnitude ``~1/|h1|``: an *extensive* ``H``
wants a probe of order ``1/N``, which is why ``0.02`` is the right ballpark
for the ~100-site clusters of the paper.  And ``alpha* > 0`` whenever the
states mixed into ``psi_G`` sit at negative energy -- essentially every
lattice model with ``E < 0`` -- while a contaminant at positive energy gives
``alpha* < 0``.  Both signs are legitimate and ``optimal_alpha`` simply
returns the root with the lower energy, as Appendix C prescribes.

Pitfalls
--------
These all come from Eq. (4) being an inversion, and none of them announces
itself in the algebra.

*The orthogonality point.*  ``chi`` is proportional to
``<psi_alpha|psi_G> = 1 + alpha h1``, which vanishes at ``alpha = -1/h1``.
That point lies *between* the two stationary points (it coincides with the
maximum, ``-1/E_0``, in the two-level limit), so a probe iterated towards
``alpha*`` from a small starting value generally has to step over it.  There
the LS state is orthogonal to the guiding state, ``chi -> 0``, and the error
on ``h2`` diverges like ``1/|chi|``.  ``measure_lanczos`` flags a small
``|chi|`` and a probe close to ``-1/h1``.

*Conditioning in the probe.*  ``chi -> 1`` with vanishing variance as
``alpha -> 0`` (``sigma_chi ~ |alpha| sigma_{e_G}/sqrt(N)``), and
``|dh2/dchi| ~ 1/alpha^2``, so ``sigma_h2 ~ 1/|alpha|`` -- one power, not two.
``sigma_h3`` really does grow like ``1/alpha^2``, because ``sigma_E`` does not
vanish with the probe.  So a small probe costs a factor ``1/|alpha|`` on
``Var_G`` and ``1/alpha^2`` on ``h3`` and hence on ``alpha*``; a probe near
``-1/h1`` costs everything.  The compromise ``|alpha h1| ~ 0.5`` is what
``lanczos_step`` uses when no ``alpha0`` is given.

*Negative sampled variance.*  For a well-optimised ``psi_G``, ``Var_G`` is
small and the *derived* ``h2`` can come out below ``h1^2``.  Then
``D(alpha) = (1 + alpha h1)^2 + alpha^2 Var_G`` acquires real zeros, (C3)
acquires poles, its infimum is ``-inf``, and neither stationary point is a
minimum -- "the root with the lower energy" would return a local *maximum*.
``optimal_alpha`` refuses (returns ``nan``) when ``h2 <= h1^2``.  Note that
this is the *only* way its ``nan`` path can be reached: the discriminant of
(7) is identically

    b^2 - 4 a c = (h3 - 3 h1 h2 + 2 h1^3)^2 + 4 (h2 - h1^2)^3
               >= 4 Var_G^3,                                              (8)

so no amount of noise in ``h3`` can make (C4) complex while ``Var_G > 0``.

*What (C5) does and does not check.*  At the self-consistent probe

    chi = 1 / (1 + alpha* E(alpha*))                                      (C5)

holds identically, and substituting (6) and (C3) gives the exact

    chi - 1/(1 + alpha E) = alpha^2 (c + b alpha + a alpha^2)
                          / [D (1 + 3 alpha h1 + 3 alpha^2 h2 + alpha^3 h3)],

whose numerator carries the very quadratic (7) that (C4) solves.  Because
``h2`` and ``h3`` are obtained by *exactly* inverting the same measured
``chi`` and ``E(alpha)``, the residual is an algebraic restatement of "the
probe is a stationary point of (C3)" and carries no information about whether
``chi``, ``E(alpha)`` or ``h1`` is biased -- a run with a 1% bias in ``chi``
reports a residual of exactly zero.  It also vanishes at the *maximum* root.
Read it as a convergence measure for the probe, nothing more.

The one genuinely independent check available is
``LanczosResult.variance_mismatch``: the ``chi``-derived ``Var_G = h2 - h1^2``
against the direct ``<H^2> - <H>^2`` that ``measure_energy`` gets from
``<|E_loc|^2>`` on the ``|psi_G|^2`` chain.  Different estimators, different
chains, same quantity; a disagreement beyond the error bars means one of them
is wrong.

Cost
----
Sampling ``|psi_alpha|^2`` evaluates ``psi_alpha`` -- and therefore one local
energy of ``psi_G`` -- at every Metropolis proposal, and measuring
``E(alpha)`` needs ``psi_alpha`` on every connected configuration, i.e. two
powers of ``H``.  Expect a cost per step of order ``n_terms`` (sampling) to
``n_terms^2`` (measurement) times a plain VMC step.  In exchange, ``<H^2>``
and ``<H^3>`` of the guiding state come out for free.

Usage
-----
Full self-consistent step (measures ``h1`` itself, picks its own probe)::

    from tachys.experimental.lanczos import lanczos_step
    key, state, wf_ls, res = lanczos_step(
        key, wf, H, state, action, N_mc, N_steps=50, n_iter=3, warmup=20)
    print(res.alpha_star, res.energy_star, res.variance_guiding)

``wf_ls`` is a plain ``WaveFunction`` at the optimal ``alpha``, so it drops
straight into ``compute_observables`` for correlation functions over the LS
state, or into ``compute_expectation`` for its energy and variance.  For the
variance extrapolation of Sorella's Figs. 6-9, the ``(E, sigma^2)`` pair must
come from the *same* wavefunction: use ``res.energy`` with
``res.variance_alpha`` (both measured at ``res.alpha``, and valid as a pair
only when ``res.converged`` is True, i.e. when the probe has reached
``alpha*``), or measure ``wf_ls`` directly -- not ``res.energy_star``, which
is the (C3) prediction at ``alpha*`` rather than a measurement.

Nesting ``lanczos_wavefunction`` twice gives ``(1 + a2 H)(1 + a1 H) psi_G``,
a two-parameter subfamily of the ``p = 2`` Krylov space (it reaches
``1 + a H + b H^2`` only for ``a^2 >= 4 b``, the real-root cases), at another
factor ``n_terms`` in cost.  It is not the general two-step Lanczos state.

Not supported
-------------
Foundation-model Hamiltonians, i.e. operators whose ``coupling`` is 2-D with
one column per Monte Carlo chain (``lattice.foundation.operators``).  Their
coupling is indexed by walker *position* in the batch, and this module has to
evaluate ``H`` inside ``apply_fn``, which tachys calls on resized and
permuted walker sub-batches (``mc_step`` passes a quarter of the chains,
``_apply_masked`` passes a boolean-partition permutation of the flattened
connected states).  The columns would silently stop lining up with the rows.
``lanczos_wavefunction`` detects and rejects such an operator; supporting it
needs the coupling carried as a per-sample field of the ``State``, so that it
permutes with the configurations.
"""
import dataclasses
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import shard_map
from jax.sharding import PartitionSpec as P

from tachys.experimental.fidelity import blocking_error
from tachys.ground_state_training import Timer
from tachys.lattice.operator.base import _OperatorMul, _OperatorSum
from tachys.lattice.operator.local_estimator import compute_expectation, local_estimator
from tachys.lattice.state_array import get_n_mc_local
from tachys.montecarlo import sample
from tachys.parallel import mesh, n_devices, rank, MASTER
from tachys.utils import _cast_floating_to


# ---------------------------------------------------------------------------
# the Lanczos-step wavefunction
# ---------------------------------------------------------------------------

def _per_chain_coupling_leaves(operator):
    """Names of operator leaves whose ``coupling`` is 2-D (one column per chain).

    See "Not supported" in the module docstring: such an operator cannot be
    evaluated inside an ``apply_fn``, because ``apply_fn`` is handed resized and
    permuted walker sub-batches whose rows no longer match the coupling columns.
    """
    if isinstance(operator, (_OperatorSum, _OperatorMul)):   # same traversal as
        return [name for sub in operator.operators               # foundation.operators._iter_leaves
                for name in _per_chain_coupling_leaves(sub)]
    coupling = getattr(operator, 'coupling', None)
    if coupling is not None and jnp.ndim(coupling) == 2:
        return [type(operator).__name__]
    return []


def lanczos_wavefunction(wf, H, alpha, optimize_mask=True, batch_expand=1):
    """``(1 + alpha H) psi_G`` as a drop-in ``WaveFunction``.

    Returns a copy of ``wf`` whose ``apply_fn`` evaluates

        log psi_alpha(x) = log1p(alpha * e_G(x)) + log psi_G(x),

    with ``e_G`` the local energy of ``wf`` (Eq. 1).  ``params``, ``dtype``
    and ``unravel_params_fn`` are those of ``wf``, so ``sample``,
    ``compute_expectation`` and ``compute_observables`` all accept the result
    unchanged -- including their casting of floating leaves to ``wf.dtype``,
    which reaches ``psi_G`` because the closure rebuilds it from the
    ``params`` it is handed rather than from the ones captured here.

    ``log1p`` is used rather than ``log(1 + .)`` because the interesting
    regime is ``|alpha e_G| << 1``; on a configuration where
    ``1 + alpha e_G < 0`` it correctly returns a complex logarithm, carrying
    the sign change of the LS state, and a configuration with
    ``alpha e_G = -1`` sits on a node and gets ``-inf``, which the Metropolis
    test rejects.

    The value of ``alpha`` is recorded on the returned closure as
    ``wf_alpha.apply_fn.lanczos_alpha`` -- there is no field on
    ``WaveFunction`` to put it in, and ``measure_lanczos`` needs to be able to
    check that a wavefunction handed to it was built at the ``alpha`` its
    algebra is about.

    Parameters
    ----------
    wf            : WaveFunction -- the guiding state ``psi_G``.
    H             : operator -- the same Hamiltonian whose LS step is taken.
                    An operator with a per-chain (2-D) ``coupling`` is rejected;
                    see "Not supported" in the module docstring.
    alpha         : float -- real; its sign is state dependent, see "The sign
                    and scale of alpha*" in the module docstring.
    optimize_mask, batch_expand
        Forwarded to ``local_estimator`` for the inner ``psi_G`` evaluation.
        The default ``batch_expand=1`` is the only value guaranteed to satisfy
        that function's divisibility assertion for *any* incoming batch size,
        and this ``apply_fn`` is called on batches whose size is chosen by the
        caller (``mc_step`` uses ``0.25 * N_mc``), so raise it only if you have
        checked the arithmetic.

    Notes
    -----
    ``alpha`` is baked into the returned closure, so every new value triggers a
    fresh JAX trace of ``sample``/``compute_expectation``.  That is a handful
    of compilations per optimisation, not per step.
    """
    alpha = float(np.real(alpha))
    bad = sorted(set(_per_chain_coupling_leaves(H)))
    if bad:
        raise ValueError(
            f"lanczos_wavefunction cannot use a Hamiltonian with a per-chain coupling "
            f"(2-D `coupling`, one column per Monte Carlo chain); offending operator "
            f"leaves: {bad}. Such a coupling is indexed by walker position in the batch, "
            f"but the Lanczos-step apply_fn is called on resized and permuted walker "
            f"sub-batches (mc_step passes 0.25*N_mc chains, _apply_masked passes a "
            f"permutation of the flattened connected states), so the columns would "
            f"silently stop matching the rows. Supporting this needs the coupling "
            f"carried as a per-sample field of the State so that it permutes with the "
            f"configurations."
        )

    def apply_fn(params, state):
        psi_G = wf.replace(params=params)
        log_amps = psi_G.apply_fn(params, state).astype(jnp.complex128)
        e_G = local_estimator(H, state, psi_G, log_amps,
                              optimize_mask=optimize_mask, batch_expand=batch_expand)
        return jnp.log1p(alpha * e_G) + log_amps

    apply_fn.lanczos_alpha = alpha
    return wf.replace(apply_fn=apply_fn)


@jax.jit
def chi_estimator(wf, state, log_amps_alpha):
    """``chi`` of Eq. (3), from the amplitude ratio ``psi_G / psi_alpha``.

    Parameters
    ----------
    wf             : WaveFunction -- the *guiding* state.
    state          : State sampled from ``|psi_alpha|^2``, batch axis sharded on 'i'.
    log_amps_alpha : (N_mc_local,) ``log psi_alpha`` on that state -- exactly
                     what ``sample`` returns for the chain it propagated.

    Returns
    -------
    chi, log_amps_G
        ``chi`` is the global (all-device) mean of ``psi_G/psi_alpha``, equal to
        ``<psi_alpha|psi_G>/<psi_alpha|psi_alpha>``; real up to Monte Carlo
        noise, and returned complex so the caller can use the imaginary part as
        a convergence diagnostic.  ``log_amps_G`` is ``log psi_G`` on the same
        configurations, returned because it costs nothing extra and makes the
        per-configuration local energy of ``psi_G`` recoverable as
        ``(exp(log_amps_alpha - log_amps_G) - 1) / alpha``.

    The ratio needs no log-shift guard: ``log psi_G - log psi_alpha =
    -log1p(alpha e_G)`` is a difference of two amplitudes of the *same*
    wavefunction family, small wherever ``|alpha e_G| << 1``, so there is no
    arbitrary relative normalisation to overflow (contrast ``fidelity``, where
    the two states are unrelated).
    """
    def _body(wf, state, log_amps_alpha):
        wf, state = _cast_floating_to((wf, state), wf.dtype)
        log_amps_G = wf.apply_fn(wf.params, state).astype(jnp.complex128)
        ratio = jnp.exp(log_amps_G - log_amps_alpha)
        chi = jax.lax.psum(jnp.mean(ratio), 'i') / n_devices
        return chi, log_amps_G

    return shard_map(
        _body, mesh=mesh,
        in_specs=(P(), P('i'), P('i')),
        out_specs=(P(), P('i')),
        check_vma=False,
    )(wf, state, log_amps_alpha)


# ---------------------------------------------------------------------------
# Appendix C algebra -- pure functions of scalars, no Monte Carlo
# ---------------------------------------------------------------------------

def lanczos_energy(alpha, h1, h2, h3):
    """``E(alpha)`` of Eq. (C3)."""
    return (h1 + 2.0 * alpha * h2 + alpha ** 2 * h3) / (1.0 + 2.0 * alpha * h1 + alpha ** 2 * h2)


def lanczos_chi(alpha, h1, h2):
    """``chi(alpha)`` of Eq. (6) -- the inverse of ``h2 = ...(chi)`` below."""
    return (1.0 + alpha * h1) / (1.0 + 2.0 * alpha * h1 + alpha ** 2 * h2)


def lanczos_moments(alpha, h1, chi, energy):
    """``(h2, h3)`` from the two sampled averages -- Eqs. (4)-(5).

    Exact inversions of ``lanczos_chi`` and ``lanczos_energy``, so
    ``lanczos_moments(a, h1, lanczos_chi(a, h1, h2), lanczos_energy(a, h1, h2, h3))``
    returns ``(h2, h3)`` back.

    Both divide by ``alpha^2``, but they do not degrade alike as the probe
    shrinks: ``chi -> 1`` with vanishing variance, so ``sigma_h2 ~ 1/|alpha|``,
    whereas ``sigma_E`` stays finite and ``sigma_h3 ~ 1/alpha^2``.  See
    "Conditioning in the probe" in the module docstring.
    """
    h2 = ((1.0 / chi - 2.0) * (1.0 + alpha * h1) + 1.0) / alpha ** 2
    h3 = (energy * (1.0 + 2.0 * alpha * h1 + alpha ** 2 * h2) - h1 - 2.0 * alpha * h2) / alpha ** 2
    return h2, h3


def stationarity_coefficients(h1, h2, h3):
    """``(a, b, c)`` of the quadratic (7) whose roots are the stationary points.

    ``a = h1 h3 - h2^2``, ``b = h3 - h1 h2``, ``c = h2 - h1^2 = Var_G``.  The
    sign of ``a`` decides whether the two roots share a sign (``a > 0``) or
    straddle zero (``a < 0``); ``c > 0`` is required for (C3) to have a minimum
    at all.
    """
    return h1 * h3 - h2 ** 2, h3 - h1 * h2, h2 - h1 ** 2


def optimal_alpha(h1, h2, h3):
    """``alpha*`` minimising (C3), and ``E(alpha*)`` -- Eq. (C4).

    ``dE/dalpha = 0`` reduces to the quadratic (7); its two roots are the
    minimum and the maximum of (C3), and the one with the lower ``E`` is
    returned.  The sign of ``alpha*`` follows the sign of ``a = h1 h3 - h2^2``
    and is state dependent -- see "The sign and scale of alpha*" in the module
    docstring; it is *not* fixed by ``dE/dalpha|_0 = 2 Var_G > 0``.

    Returns
    -------
    alpha_star, energy_star, roots
        ``roots`` is the pair as computed.  All three are ``nan`` when the
        inputs cannot define a minimum, which by Eq. (8) happens in exactly one
        circumstance: ``h2 <= h1^2``, i.e. a derived variance that has come out
        zero or negative.  Then ``D(alpha) = (1 + alpha h1)^2 + alpha^2 Var_G``
        has real zeros, (C3) has poles and an infimum of ``-inf``, and the
        lower-energy root is a local *maximum*; returning it would build a
        wavefunction sitting on a pole of the norm.  A negative derived
        variance is the ordinary signature of a probe too small (or statistics
        too short) for Eq. (4) on a well-optimised ``psi_G`` -- not of a large
        ``h3`` error, which Eq. (8) shows can never make (C4) complex.
    """
    h1, h2, h3 = float(np.real(h1)), float(np.real(h2)), float(np.real(h3))
    a, b, c = stationarity_coefficients(h1, h2, h3)

    nan3 = (float('nan'), float('nan'), (float('nan'), float('nan')))
    if not all(np.isfinite([a, b, c])) or c <= 0.0:
        return nan3

    if a == 0.0:                                  # degenerate: linear in alpha
        if b == 0.0:
            return nan3
        roots = (-c / b, float('nan'))
    else:
        disc = b * b - 4.0 * a * c                # >= 4 c^3 > 0 by Eq. (8)
        if disc < 0.0:                            # unreachable for c > 0; kept honest
            return nan3
        sq = float(np.sqrt(disc))
        # Take the branch of the quadratic formula free of cancellation, then
        # the partner from Vieta (alpha_+ alpha_- = c/a); the textbook form
        # loses precision in the small root once |4ac| << b^2, which is the
        # near-eigenstate limit where all of a, b, c go to zero together.
        r1 = (-b - sq) / (2.0 * a) if b >= 0.0 else (-b + sq) / (2.0 * a)
        r2 = c / (a * r1) if r1 != 0.0 else -b / a
        roots = (r1, r2)

    energies = [lanczos_energy(r, h1, h2, h3) if np.isfinite(r) else float('nan')
                for r in roots]
    if not np.any(np.isfinite(energies)):
        return nan3
    best = int(np.nanargmin(energies))
    return roots[best], energies[best], roots


def c5_residual(alpha, chi, energy):
    """``chi - 1/(1 + alpha E(alpha))`` -- Eq. (C5), zero at a stationary point.

    Evaluate it with the *measured* ``chi`` and ``E(alpha)`` at the probe
    ``alpha``: it then measures how far the probe is from a root of the
    stationarity quadratic (7).  That is all it measures.  Substituting (6) and
    (C3) gives it exactly as ``alpha^2 (c + b alpha + a alpha^2) / [D (1 + 3
    alpha h1 + 3 alpha^2 h2 + alpha^3 h3)]``, and since ``h2`` and ``h3`` come
    from exactly inverting the same ``chi`` and ``E(alpha)``, the residual is
    an algebraic restatement of "the probe is stationary" -- it stays exactly
    zero under a bias in ``chi``, ``E(alpha)`` or ``h1``, and it also vanishes
    at the *maximum* root.  It is a convergence measure for the probe, not a
    check on the estimators; for that use ``LanczosResult.variance_mismatch``.
    """
    return chi - 1.0 / (1.0 + alpha * energy)


# ---------------------------------------------------------------------------
# error bars
# ---------------------------------------------------------------------------

def _block_means(x, n_blocks):
    """Contiguous block means of a per-step series, dropping the tail remainder."""
    x = np.asarray(x, dtype=float).ravel()
    n_blocks = int(min(n_blocks, x.size))
    if n_blocks < 2:
        return None
    b = x.size // n_blocks
    return x[:b * n_blocks].reshape(n_blocks, b).mean(axis=1)


def _jackknife(alpha, h1, h1_err, energy_series, chi_series, n_blocks=8):
    """Blocked-jackknife errors on the quantities derived from (chi, E(alpha)).

    ``h2``, ``h3``, ``alpha*`` and ``E(alpha*)`` are strongly non-linear in the
    two sampled averages, so propagating error bars analytically is both
    tedious and wrong in the tails.  Instead the per-step series are grouped
    into ``n_blocks`` contiguous blocks -- which must be long enough to absorb
    the Markov chain's autocorrelation; ``n_blocks`` is a keyword of both
    ``measure_lanczos`` and ``lanczos_step`` for that reason -- and every
    derived quantity is recomputed with one block left out, giving the usual
    jackknife estimate

        err^2 = (B - 1)/B * sum_i (x_i - mean_j x_j)^2.

    ``h1`` comes from a different measurement, so its uncertainty cannot enter
    the same resampling and is added in quadrature from a sensitivity.  That
    sensitivity is taken as a *derivative* -- a central difference over a step
    small enough to stay linear, rescaled to ``h1_err`` -- and not as a secant
    across the whole ``h1 +- h1_err`` interval: ``h1 -> alpha*`` is non-linear
    and has a pole where ``a = h1 h3 - h2^2`` changes sign, so a wide secant
    can wrap around the pole and come back *smaller*, reporting a confident
    error bar exactly where there is none.  When the interval does span that
    pole, ``alpha_star`` and ``energy_star`` are given ``inf`` rather than a
    finite bar: the data do not bound them.

    Errors are returned as ``nan`` when there are too few blocks to resample.
    """
    keys = ('h2', 'h3', 'alpha_star', 'energy_star')
    unbounded = ('alpha_star', 'energy_star')

    def _derive(h1_, chi_, e_):
        h2, h3 = lanczos_moments(alpha, h1_, chi_, e_)
        a_s, e_s, _ = optimal_alpha(h1_, h2, h3)
        return dict(h2=float(np.real(h2)), h3=float(np.real(h3)),
                    alpha_star=a_s, energy_star=e_s)

    def _leading_coeff(h1_, chi_, e_):
        h2, h3 = lanczos_moments(alpha, h1_, chi_, e_)
        return float(np.real(stationarity_coefficients(h1_, h2, h3)[0]))

    e_blocks = _block_means(np.real(energy_series), n_blocks)
    c_blocks = _block_means(np.real(chi_series), n_blocks)
    if e_blocks is None or c_blocks is None:
        return {k: float('nan') for k in keys}

    B = e_blocks.size
    samples = [_derive(h1, c_blocks[np.arange(B) != i].mean(), e_blocks[np.arange(B) != i].mean())
               for i in range(B)]

    err = {}
    for k in keys:
        v = np.array([s[k] for s in samples], dtype=float)
        err[k] = (float(np.sqrt((B - 1) / B * np.sum((v - v.mean()) ** 2)))
                  if np.all(np.isfinite(v)) else float('nan'))

    if h1_err and np.isfinite(h1_err) and h1_err > 0:
        chi_m, e_m = float(np.real(np.mean(chi_series))), float(np.real(np.mean(energy_series)))
        step = min(h1_err, 1e-4 * max(abs(h1), 1.0))
        plus, minus = _derive(h1 + step, chi_m, e_m), _derive(h1 - step, chi_m, e_m)
        a_hi, a_lo = _leading_coeff(h1 + h1_err, chi_m, e_m), _leading_coeff(h1 - h1_err, chi_m, e_m)
        spans_pole = not (np.isfinite(a_hi) and np.isfinite(a_lo)) or np.sign(a_hi) != np.sign(a_lo)
        for k in keys:
            if k in unbounded and spans_pole:
                err[k] = float('inf')
                continue
            d = 0.5 * abs(plus[k] - minus[k]) * (h1_err / step)
            if not np.isfinite(d):
                err[k] = float('inf') if k in unbounded else float('nan')
            elif np.isfinite(err[k]):
                err[k] = float(np.hypot(err[k], d))
    return err


# ---------------------------------------------------------------------------
# energy of a (possibly Lanczos) wavefunction
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class EnergyResult:
    """``<H>`` and ``<H^2> - <H>^2`` of one wavefunction along one Markov chain."""
    energy: float
    energy_err: float
    variance: float
    variance_err: float
    imag: float
    acceptance: float
    history: dict


def measure_energy(key, wf, H, state, action, N_mc, N_steps, nsweeps=1, warmup=0,
                   log_every=1, label="E"):
    """Sample ``|psi|^2`` and average the local energy -- ``h1`` when ``wf`` is ``psi_G``.

    ``wf`` is never modified.  ``variance`` is ``<H^2> - <H>^2`` from
    ``compute_expectation``'s second moment, i.e. ``h2 - h1^2`` when ``wf`` is
    the guiding state -- obtained here the expensive way, from ``<|E_loc|^2>``,
    whereas the Lanczos machinery below gets the same number out of ``chi``.
    It is kept because it costs nothing on top of the energy and because the
    comparison of the two is the module's only non-tautological check on the
    ``chi`` estimator; ``lanczos_step`` carries it through to
    ``LanczosResult.variance_direct`` and differences them there.

    Returns
    -------
    key, state, EnergyResult
    """
    if N_steps < 1:
        raise ValueError(f"N_steps must be at least 1, got {N_steps}.")
    n = get_n_mc_local(state)
    if n != N_mc:
        raise ValueError(
            f"state carries {n} chains but N_mc={N_mc}; montecarlo.sample draws one key "
            f"per chain, so they must agree."
        )

    history = {k: [] for k in ("energy", "e2", "imag", "acceptance")}
    header = (f"{'step':>5} │ {label:>20} │ {'var':>11} │ {'accept':>7} │ "
              f"{'t_mc':>6} │ {'t_exp':>6}")
    if log_every and rank == MASTER:
        print(header, flush=True)
        print("─" * len(header), flush=True)

    for step in range(-warmup, N_steps):
        key, subkey = jax.random.split(key)
        with Timer() as t_mc:
            state, log_amps, acceptance = sample(
                nsweeps, state, action, jax.random.split(subkey, N_mc), wf)
            jax.block_until_ready((state, log_amps))
        if step < 0:
            continue
        with Timer() as t_exp:
            _, e_mean, e2_mean = compute_expectation(H, wf, state, log_amps)
            jax.block_until_ready(e2_mean)

        history["energy"].append(float(jnp.real(e_mean)))
        history["e2"].append(float(jnp.real(e2_mean)))
        history["imag"].append(float(jnp.imag(e_mean)))
        history["acceptance"].append(float(jnp.mean(acceptance)))

        if log_every and rank == MASTER and (step % log_every == 0 or step == N_steps - 1):
            print(f"{step:5d} │ {history['energy'][-1]:20.12f} │ "
                  f"{history['e2'][-1] - history['energy'][-1] ** 2:11.4e} │ "
                  f"{history['acceptance'][-1]:7.3f} │ "
                  f"{t_mc.elapsed:6.2f} │ {t_exp.elapsed:6.2f}", flush=True)

    history = {k: np.asarray(v) for k, v in history.items()}
    energy = float(history["energy"].mean())
    result = EnergyResult(
        energy=energy,
        energy_err=blocking_error(history["energy"]),
        # <H^2> - <H>^2 from the averaged moments, not the mean of per-step
        # variances: the two differ by the (negligible, but nonzero) spread of
        # the per-step energies, and the former is the quantity wanted. The
        # error bar does come from the per-step series, which is what there is.
        variance=float(history["e2"].mean()) - energy ** 2,
        variance_err=blocking_error(history["e2"] - history["energy"] ** 2),
        imag=float(history["imag"].mean()),
        acceptance=float(history["acceptance"].mean()),
        history=history,
    )
    return key, state, result


# ---------------------------------------------------------------------------
# one probe of the Lanczos step
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class LanczosResult:
    """Everything one probe ``alpha`` yields.  Energies are totals, not per site.

    alpha            : the probe used.
    h1, h2, h3       : moments ``<H^n>_{psi_G}``; ``h1`` measured, ``h2``/``h3``
                       from Eqs. (4)-(5).
    energy           : measured ``E(alpha)`` -- variational, ``>= E_0``.
    chi              : measured ``chi`` of Eq. (3).
    variance_alpha   : ``<H^2> - <H>^2`` of ``psi_alpha`` at *this probe*.  With
                       ``energy`` (not ``energy_star``) it is the ``(E, sigma^2)``
                       pair for a variance extrapolation, and it describes the
                       LS state at ``alpha*`` only once the probe has converged
                       there -- see ``converged``.
    variance_guiding : ``h2 - h1^2``, the guiding state's variance, recovered
                       from ``chi`` without ever applying ``H`` twice.
    variance_direct  : the same quantity measured independently as
                       ``<|E_loc|^2> - <E_loc>^2`` on the ``|psi_G|^2`` chain, or
                       ``nan`` when it was not supplied.
    variance_mismatch: ``variance_guiding - variance_direct``.  The module's one
                       genuinely independent check: two different estimators on
                       two different chains for one number.  A disagreement well
                       outside the error bars means ``chi`` (or ``h1``) is wrong.
    alpha_star       : the minimiser of (C3), Eq. (C4).
    energy_star      : ``E(alpha*)`` predicted by (C3) -- a prediction, not a
                       measurement; re-probe at ``alpha_star`` to confirm it.
    c5_residual      : Eq. (C5) residual at the probe.  Measures the probe's
                       distance from a stationary point of (C3) and nothing
                       else; see ``c5_residual``.
    converged        : set by ``lanczos_step`` when the probe stopped moving, so
                       that this probe *is* ``alpha*``.  False on a bare
                       ``measure_lanczos``.
    imag_energy, imag_chi : discarded imaginary parts -- pure noise; if either is
                       not small compared to its real part the run is not converged.
    history          : per-step arrays ``energy``, ``chi``, ``e2``, ``acceptance``,
                       ``imag_energy``, ``imag_chi``.
    """
    alpha: float
    h1: float
    h1_err: float
    h2: float
    h2_err: float
    h3: float
    h3_err: float
    energy: float
    energy_err: float
    chi: float
    chi_err: float
    variance_alpha: float
    variance_alpha_err: float
    variance_guiding: float
    variance_direct: float
    variance_mismatch: float
    alpha_star: float
    alpha_star_err: float
    energy_star: float
    energy_star_err: float
    c5_residual: float
    imag_energy: float
    imag_chi: float
    acceptance: float
    roots: tuple
    history: dict
    converged: bool = False


def measure_lanczos(key, wf, H, state, action, N_mc, N_steps, alpha, h1, h1_err=0.0,
                    nsweeps=1, warmup=0, log_every=1, n_blocks=8,
                    optimize_mask=True, batch_expand=1, wf_alpha=None,
                    variance_direct=float('nan')):
    """Probe the LS state at one ``alpha``: measure ``E(alpha)`` and ``chi``, solve (C4).

    The chain is propagated under ``|psi_alpha|^2`` (not ``|psi_G|^2``), which
    is what both (2) and (3) require, so ``state`` must be equilibrated for
    ``psi_alpha`` -- pass ``warmup > 0`` when it comes from a ``psi_G`` chain or
    from a previous, different ``alpha``.

    Parameters
    ----------
    wf       : WaveFunction -- the guiding state ``psi_G``, unmodified.
    H        : operator.
    state    : State -- starting configurations, batch axis sharded.
    alpha    : float -- the probe; see "Conditioning in the probe" and "The sign
               and scale of alpha*" in the module docstring.
    h1       : float -- ``<H>_{psi_G}``, from ``measure_energy`` on ``wf``.
    h1_err   : float -- its error bar, propagated into the derived quantities.
    n_blocks : int -- jackknife blocks over the ``N_steps`` measurements; they
               must be longer than the autocorrelation time.
    wf_alpha : WaveFunction, optional -- a prebuilt ``lanczos_wavefunction(wf, H,
               alpha)`` to reuse, avoiding one JAX recompilation.  It must have
               been built at this same ``alpha``, which is checked against the
               ``lanczos_alpha`` that ``lanczos_wavefunction`` records on the
               closure: the chain would otherwise be sampled from one LS state
               while Eqs. (4)-(5) were solved for another, silently.
    variance_direct : float, optional -- an independently measured
               ``<H^2> - <H>^2`` of ``psi_G`` (``EnergyResult.variance``) to
               difference against the ``chi``-derived one.

    Returns
    -------
    key, state, LanczosResult
    """
    alpha = float(np.real(alpha))
    if alpha == 0.0:
        raise ValueError("alpha must be non-zero: Eqs. (4)-(5) divide by alpha^2.")
    if N_steps < 1:
        raise ValueError(f"N_steps must be at least 1, got {N_steps}.")
    n = get_n_mc_local(state)
    if n != N_mc:
        raise ValueError(f"state carries {n} chains but N_mc={N_mc}.")

    if wf_alpha is None:
        wf_alpha = lanczos_wavefunction(wf, H, alpha,
                                        optimize_mask=optimize_mask, batch_expand=batch_expand)
    else:
        baked = getattr(wf_alpha.apply_fn, 'lanczos_alpha', None)
        if baked is None:
            raise ValueError(
                "wf_alpha must come from lanczos_wavefunction (which records the alpha it "
                "was built at on the returned closure); a wavefunction of unknown alpha "
                "cannot be checked against the alpha this function solves (C4) for."
            )
        if float(baked) != alpha:
            raise ValueError(
                f"wf_alpha was built at alpha={baked!r} but this call solves Eqs. (4)-(5) "
                f"for alpha={alpha!r}. Sampling one LS state and inverting for another "
                f"silently corrupts h2, h3 and alpha*."
            )

    history = {k: [] for k in ("energy", "chi", "e2", "imag_energy", "imag_chi", "acceptance")}
    header = (f"{'step':>5} │ {'E(alpha)':>20} │ {'chi':>12} │ {'accept':>7} │ "
              f"{'t_mc':>6} │ {'t_exp':>6} │ {'t_chi':>6}")
    if log_every and rank == MASTER:
        print(header, flush=True)
        print("─" * len(header), flush=True)

    for step in range(-warmup, N_steps):
        key, subkey = jax.random.split(key)
        with Timer() as t_mc:
            state, log_amps_alpha, acceptance = sample(
                nsweeps, state, action, jax.random.split(subkey, N_mc), wf_alpha)
            jax.block_until_ready((state, log_amps_alpha))
        if step < 0:
            continue

        with Timer() as t_exp:
            # Local energy of psi_alpha on its own samples: Eq. (2), plus its
            # second moment, which is <H^2>_{psi_alpha} exactly (Hermitian H).
            _, e_mean, e2_mean = compute_expectation(H, wf_alpha, state, log_amps_alpha)
            jax.block_until_ready(e2_mean)
        with Timer() as t_chi:
            chi, _ = chi_estimator(wf, state, log_amps_alpha)
            jax.block_until_ready(chi)

        history["energy"].append(float(jnp.real(e_mean)))
        history["e2"].append(float(jnp.real(e2_mean)))
        history["chi"].append(float(jnp.real(chi)))
        history["imag_energy"].append(float(jnp.imag(e_mean)))
        history["imag_chi"].append(float(jnp.imag(chi)))
        history["acceptance"].append(float(jnp.mean(acceptance)))

        if log_every and rank == MASTER and (step % log_every == 0 or step == N_steps - 1):
            print(f"{step:5d} │ {history['energy'][-1]:20.12f} │ {history['chi'][-1]:12.8f} │ "
                  f"{history['acceptance'][-1]:7.3f} │ {t_mc.elapsed:6.2f} │ "
                  f"{t_exp.elapsed:6.2f} │ {t_chi.elapsed:6.2f}", flush=True)

    history = {k: np.asarray(v) for k, v in history.items()}
    energy = float(history["energy"].mean())
    chi = float(history["chi"].mean())
    h1 = float(np.real(h1))

    h2, h3 = lanczos_moments(alpha, h1, chi, energy)
    alpha_star, energy_star, roots = optimal_alpha(h1, h2, h3)
    err = _jackknife(alpha, h1, h1_err, history["energy"], history["chi"], n_blocks=n_blocks)
    variance_guiding = float(h2) - h1 ** 2
    variance_direct = float(np.real(variance_direct))

    return key, state, LanczosResult(
        alpha=alpha,
        h1=h1, h1_err=float(h1_err),
        h2=float(h2), h2_err=err["h2"],
        h3=float(h3), h3_err=err["h3"],
        energy=energy, energy_err=blocking_error(history["energy"]),
        chi=chi, chi_err=blocking_error(history["chi"]),
        variance_alpha=float(history["e2"].mean()) - energy ** 2,
        variance_alpha_err=blocking_error(history["e2"] - history["energy"] ** 2),
        variance_guiding=variance_guiding,
        variance_direct=variance_direct,
        variance_mismatch=variance_guiding - variance_direct,
        alpha_star=alpha_star, alpha_star_err=err["alpha_star"],
        energy_star=energy_star, energy_star_err=err["energy_star"],
        c5_residual=float(np.real(c5_residual(alpha, chi, energy))),
        imag_energy=float(history["imag_energy"].mean()),
        imag_chi=float(history["imag_chi"].mean()),
        acceptance=float(history["acceptance"].mean()),
        roots=roots,
        history=history,
    )


# ---------------------------------------------------------------------------
# the self-consistent step
# ---------------------------------------------------------------------------

def _probe_warnings(result):
    """Diagnostics that the algebra of Appendix C cannot raise for itself."""
    msgs = []
    r = result
    if r.variance_guiding <= 0.0:
        msgs.append(
            f"the chi-derived variance came out {r.variance_guiding:.3e} <= 0, so (C3) has "
            f"poles and no minimum, and alpha* is nan by construction (Eq. 8). Either the "
            f"probe is too small for Eq. (4) -- sigma_h2 ~ 1/|alpha| -- or N_steps is too "
            f"short for this psi_G's variance.")
    if abs(1.0 + r.alpha * r.h1) < 0.1:
        msgs.append(
            f"the probe sits close to the orthogonality point alpha = -1/h1 = "
            f"{-1.0 / r.h1:.6g} (1 + alpha h1 = {1.0 + r.alpha * r.h1:.3e}), where "
            f"<psi_alpha|psi_G> vanishes, chi -> 0 and the error on h2 diverges like "
            f"1/|chi|. Move the probe.")
    if abs(r.chi) < 0.05:
        msgs.append(
            f"|chi| = {abs(r.chi):.3e} is small, so h2 (Eq. 4) is ill-conditioned: its "
            f"error grows like 1/|chi|.")
    if np.isfinite(r.energy_star) and r.energy_star > r.h1:
        msgs.append(
            f"E(alpha*) = {r.energy_star:.8g} came out above h1 = {r.h1:.8g}, which is "
            f"impossible for the exact minimum of (C3) since alpha = 0 is in the family. "
            f"The moments are dominated by noise.")
    if np.isfinite(r.energy_star) and r.variance_guiding > 0:
        gain, spread = r.h1 - r.energy_star, np.sqrt(r.variance_guiding)
        if gain > spread:
            msgs.append(
                f"the predicted gain h1 - E(alpha*) = {gain:.4g} exceeds sqrt(Var_G) = "
                f"{spread:.4g}. A single Lanczos step cannot plausibly recover more energy "
                f"than the guiding state's own energy spread, so E(alpha*) is probably an "
                f"artefact of an ill-conditioned Eq. (4)/(5) inversion; check that "
                f"E(alpha*) is not below any known bound on E_0.")
    if np.isfinite(r.variance_mismatch):
        tol = 3.0 * max(abs(r.h2_err), abs(r.variance_direct) * 1e-3, 1e-12)
        if abs(r.variance_mismatch) > tol:
            msgs.append(
                f"the chi-derived Var_G = {r.variance_guiding:.6g} disagrees with the direct "
                f"<H^2>-<H>^2 = {r.variance_direct:.6g} (difference "
                f"{r.variance_mismatch:.3e}, h2 error bar {r.h2_err:.3e}). These are "
                f"independent estimators of one number on two chains; a real disagreement "
                f"means chi or h1 is biased -- equilibration is the usual cause.")
    return msgs


def lanczos_step(key, wf, H, state, action, N_mc, N_steps, alpha0=None, n_iter=3,
                 h1=None, h1_err=0.0, h1_steps=None, variance_direct=float('nan'),
                 nsweeps=1, warmup=5, tol=1e-3, log_every=0, n_blocks=8,
                 optimize_mask=True, batch_expand=1):
    """One Lanczos step, with the probe ``alpha`` iterated to ``alpha*``.

    Measures ``h1`` on ``|psi_G|^2`` (unless given), then repeats
    ``measure_lanczos`` with ``alpha <- alpha*`` until the probe stops moving
    (``|alpha* - alpha| <= tol |alpha*|``) or ``n_iter`` rounds are spent.  Two
    or three rounds is what Appendix C reports as sufficient.

    Parameters
    ----------
    key      : jax.random.key
    wf       : WaveFunction -- guiding state, never modified.
    H        : operator.
    state    : State -- batch axis sharded; reused (and returned) for every chain.
    action   : _BaseAction -- proposal for both the ``psi_G`` and ``psi_alpha`` chains.
    N_mc     : int -- Markov chains.
    N_steps  : int -- measurements per probe.
    alpha0   : float, optional -- first probe.  Defaults to ``-0.5/h1``, i.e.
               ``|alpha h1| = 0.5``: far enough from zero that Eq. (4)-(5) are
               not swamped (``sigma_h2 ~ 1/|alpha|``, ``sigma_h3 ~ 1/alpha^2``)
               and a factor two short of the orthogonality point ``-1/h1`` where
               ``chi`` vanishes.  Its *sign* is a conditioning choice, not a
               prediction: ``alpha*`` follows the sign of ``h1 h3 - h2^2`` and
               (C4) recovers it from a probe on either side.
    n_iter   : int -- maximum probes; at least 1.
    h1, h1_err : float, optional -- a known ``<H>_{psi_G}`` and its error, to skip
               the guiding-state run.  Pass ``variance_direct`` along with them
               to keep the independent variance cross-check alive.
    h1_steps : int, optional -- measurements for the ``h1`` run (default ``N_steps``).
    variance_direct : float, optional -- see ``measure_lanczos``; measured
               automatically when ``h1`` is not supplied.
    warmup   : int -- equilibration steps discarded at the start of *every* chain,
               including after each change of ``alpha``.  The default 5 is a
               floor, not a substitute for knowing the autocorrelation time: the
               first probe inherits a ``|psi_G|^2``-distributed state and each
               later one a chain equilibrated for a *different* ``alpha``, and a
               residual bias in ``chi`` is amplified into ``Var_G`` by
               ``|dh2/dchi| ~ 1/alpha^2``.  Because the same chain is carried
               from the ``h1`` run into the probes, ``h1`` and ``(chi, E)`` are
               not strictly independent either, which the quadrature addition in
               ``_jackknife`` assumes; ``warmup`` is what weakens that too.
    tol      : float -- relative convergence threshold on the probe.
    log_every: int -- per-step logging inside each probe (0 = only the summary table).

    Returns
    -------
    key, state, wf_alpha, result
        ``wf_alpha`` is ``lanczos_wavefunction(wf, H, result.alpha_star)`` -- the
        optimal LS state, ready for ``compute_observables``.  ``result`` is the
        ``LanczosResult`` of the *last probe actually measured*, with
        ``converged`` set when that probe had reached ``alpha*``; its
        ``energy_star``/``alpha_star`` are the prediction the returned
        wavefunction is built from.  ``result.history['probes']`` lists one dict
        per probe.
    """
    if n_iter < 1:
        raise ValueError(f"n_iter must be at least 1, got {n_iter}.")
    if N_steps < 1:
        raise ValueError(f"N_steps must be at least 1, got {N_steps}.")
    Ns = state.Ns
    if h1_steps is None:
        h1_steps = N_steps

    print("\n--- Lanczos step (Sorella cond-mat/0009149, App. C) ---")
    print(f"N_mc: {N_mc}    N_steps: {N_steps}    warmup: {warmup}    nsweeps: {nsweeps}")
    print(f"n_iter: {n_iter}    tol: {tol:.1e}    n_blocks: {n_blocks}    Ns: {Ns}")
    if warmup == 0:
        print("NOTE: warmup=0 -- the first probe starts measuring |psi_alpha|^2 averages "
              "from a |psi_G|^2-distributed chain, and each later probe from a chain "
              "equilibrated for the previous alpha. Any residual bias in chi is amplified "
              "into Var_G; LanczosResult.variance_mismatch is the diagnostic.")
    print("-" * 72)

    if h1 is None:
        print(f"\n[h1] sampling |psi_G|^2 for <H>_G  ({h1_steps} steps)")
        key, state, e_res = measure_energy(
            key, wf, H, state, action, N_mc, h1_steps,
            nsweeps=nsweeps, warmup=warmup, log_every=log_every, label="h1")
        h1, h1_err, variance_direct = e_res.energy, e_res.energy_err, e_res.variance
        print(f"[h1] h1/Ns = {h1 / Ns:.12f} ± {h1_err / Ns:.2e}"
              f"    var_G/Ns (direct) = {e_res.variance / Ns:.4e} ± {e_res.variance_err / Ns:.1e}"
              f"    accept = {e_res.acceptance:.3f}")
    else:
        h1, h1_err = float(np.real(h1)), float(h1_err)
        print(f"\n[h1] given: h1/Ns = {h1 / Ns:.12f} ± {h1_err / Ns:.2e}")

    if alpha0 is None:
        if h1 == 0.0:
            raise ValueError(
                "cannot pick a default alpha0 from h1 = 0 (the default is -0.5/h1); "
                "pass alpha0 explicitly."
            )
        alpha0 = -0.5 / h1
        print(f"[probe] alpha0 = -0.5/h1 = {alpha0:.6g}"
              f"    (orthogonality point -1/h1 = {-1.0 / h1:.6g})")

    header = (f"{'probe':>5} │ {'alpha':>12} │ {'E(alpha)/Ns':>18} │ {'var_a/Ns':>10} │ "
              f"{'chi':>11} │ {'h2':>13} │ {'h3':>15} │ {'var_G/Ns':>10} │ "
              f"{'alpha*':>12} │ {'E*/Ns':>18} │ {'C5 res':>9} │ {'t (s)':>7}")
    rule = "─" * len(header)
    print("\n" + header, flush=True)
    print(rule, flush=True)

    alpha = float(np.real(alpha0))
    probes, result, converged = [], None, False
    for it in range(n_iter):
        t0 = time.perf_counter()
        key, state, result = measure_lanczos(
            key, wf, H, state, action, N_mc, N_steps, alpha, h1, h1_err=h1_err,
            nsweeps=nsweeps, warmup=warmup, log_every=log_every, n_blocks=n_blocks,
            optimize_mask=optimize_mask, batch_expand=batch_expand,
            variance_direct=variance_direct)
        dt = time.perf_counter() - t0
        probes.append(dict(alpha=result.alpha, energy=result.energy, chi=result.chi,
                           h2=result.h2, h3=result.h3, alpha_star=result.alpha_star,
                           energy_star=result.energy_star, c5_residual=result.c5_residual,
                           variance_alpha=result.variance_alpha,
                           variance_guiding=result.variance_guiding))
        print(f"{it:5d} │ {result.alpha:12.6g} │ {result.energy / Ns:18.12f} │ "
              f"{result.variance_alpha / Ns:10.3e} │ {result.chi:11.7f} │ "
              f"{result.h2:13.6g} │ {result.h3:15.8g} │ "
              f"{result.variance_guiding / Ns:10.3e} │ {result.alpha_star:12.6g} │ "
              f"{result.energy_star / Ns:18.12f} │ {result.c5_residual:9.2e} │ {dt:7.1f}",
              flush=True)
        for msg in _probe_warnings(result):
            print(f"  ! {msg}", flush=True)

        if not np.isfinite(result.alpha_star):
            print("  alpha* is nan: by Eq. (8) that means the chi-derived variance came out "
                  "<= 0, so (C3) has no minimum. Keeping the previous alpha.", flush=True)
            break
        converged = abs(result.alpha_star - alpha) <= tol * abs(result.alpha_star)
        alpha = result.alpha_star
        if converged:
            print(f"  probe converged: |alpha* - alpha| <= {tol:.1e} |alpha*|", flush=True)
            break

    result.converged = bool(converged)
    print(rule, flush=True)
    alpha_final = result.alpha_star if np.isfinite(result.alpha_star) else result.alpha
    wf_alpha = lanczos_wavefunction(wf, H, alpha_final,
                                    optimize_mask=optimize_mask, batch_expand=batch_expand)

    print(f"h1/Ns            = {result.h1 / Ns:.12f} ± {result.h1_err / Ns:.2e}")
    print(f"h2               = {result.h2:.8g} ± {result.h2_err:.2e}")
    print(f"h3               = {result.h3:.8g} ± {result.h3_err:.2e}")
    print(f"var_G/Ns (chi)   = {result.variance_guiding / Ns:.6e}   (= (h2 - h1^2)/Ns)")
    print(f"var_G/Ns (direct)= {result.variance_direct / Ns:.6e}"
          f"   mismatch = {result.variance_mismatch / Ns:.2e}   (independent check)")
    print(f"alpha*           = {result.alpha_star:.8g} ± {result.alpha_star_err:.2e}")
    print(f"E(alpha*)/Ns     = {result.energy_star / Ns:.12f} ± {result.energy_star_err / Ns:.2e}"
          f"   (predicted by (C3), not measured)")
    print(f"gain vs h1       = {(result.energy_star - result.h1) / Ns:.3e} per site")
    print(f"C5 residual      = {result.c5_residual:.3e}   (probe vs stationary point only)")
    print(f"Im E, Im chi     = {result.imag_energy:.2e}, {result.imag_chi:.2e}   (noise)")
    print(f"converged        = {result.converged}"
          f"    (probe {result.alpha:.6g} vs alpha* {result.alpha_star:.6g})")
    print(f"variance extrapolation pair at alpha = {result.alpha:.6g}:"
          f"  E/Ns = {result.energy / Ns:.12f} ± {result.energy_err / Ns:.2e},"
          f"  var/Ns = {result.variance_alpha / Ns:.6e} ± {result.variance_alpha_err / Ns:.1e}")
    if not result.converged:
        print("WARNING: the probe never reached alpha*, so the (E, var) pair above describes "
              "the LS state at the last probe, NOT at alpha*. For an extrapolation point at "
              "alpha*, measure the returned wavefunction (or raise n_iter).", flush=True)
    for msg in _probe_warnings(result):
        print(f"WARNING: {msg}", flush=True)

    result.history["probes"] = probes
    return key, state, wf_alpha, result
