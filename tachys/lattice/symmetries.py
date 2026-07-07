import jax
import jax.numpy as jnp
import numpy as np

from tachys.lattice.fermions.fermion_state import FermionState


def singlet_symm(wf_apply):
    """Wrap a wavefunction to enforce global spin-flip symmetry (Z2).

    Projects onto the even sector under global spin flip σ → -σ by computing
    the symmetrized log-amplitude:

        log[ψ(σ) + ψ(-σ)] = f(σ) + log[1 + exp(f(-σ) - f(σ))]

    where f(σ) = log ψ(σ). This corresponds to a singlet-like (S_z = 0)
    symmetrization in the spin basis.

    Args:
        wf_apply: Callable (params, lattice) → log-amplitude f(σ).

    Returns:
        Wrapped callable with the same signature returning the symmetrized
        log-amplitude.
    """
    def singlet_wf_apply(params, lattice):
        f_sigma = wf_apply(params, lattice)
        f_minus_sigma = wf_apply(params, lattice.replace(spins = -lattice.spins))
        dF = f_minus_sigma - f_sigma

        return f_sigma + jnp.log(1. + jnp.exp(dF))
    return singlet_wf_apply

def time_reversal(apply_fn):
    """Wrap a wavefunction to enforce time-reversal symmetry.

    Symmetrizes the log-amplitude under complex conjugation (time reversal
    maps ψ → ψ*), producing a real-valued wavefunction:

        log[ψ(σ) + ψ*(σ)] = log[2 Re ψ(σ)]
                           = log_amps + log[1 + exp(-2i · Im(log_amps))]

    The result is a log-amplitude whose imaginary part encodes the sign of the
    wavefunction: 0 when Re ψ > 0, π when Re ψ < 0.

    Args:
        apply_fn: Callable (params, state) → complex log-amplitude.

    Returns:
        Wrapped callable with the same signature returning the time-reversal-
        symmetrized log-amplitude.
    """
    def time_reversal_apply_fn(params, state):
        log_amps = apply_fn(params, state)
        return log_amps + jnp.log(1. + jnp.exp(-2j*log_amps.imag))
    return time_reversal_apply_fn


def invert_perm(perms):
    """
    Inverse of a site permutation, or a batch of them (any leading shape,
    last axis = Ns). Every row is an honest bijection of {0,...,Ns-1}, so the
    inverse is simply the argsort - no reference row or row-matching needed.
    """
    return np.argsort(perms, axis=-1)


def expand_perm(perm, n_bands):
    """
    Widen an Ns-wide site permutation (or a batch, shape (..., Ns)) to act on
    a State.config of width n_bands*Ns, by applying the SAME geometric
    permutation independently inside each contiguous Ns-band slice
    (config[..., b*Ns:(b+1)*Ns]). n_bands=1 is a no-op (covers SpinState).
    """
    Ns = perm.shape[-1]
    return np.concatenate([perm + b * Ns for b in range(n_bands)], axis=-1)


def fermionic_sign(perm):
    """
    Sign of `perm` via inversion counting: (-1)^{#{i<j : perm[i]>perm[j]}}.
    `perm` may contain the Ns+1 sentinel in trailing slots (see
    sign_permutation) - sentinel-vs-sentinel and sentinel-vs-real pairs never
    register as inversions since the sentinel exceeds every real value, so
    the padding is inert. Single sample; vmap at the call site for a batch.
    """
    n = perm.shape[-1]
    i, j = jnp.triu_indices(n, k=1)
    inversions = jnp.sum(perm[i] > perm[j])
    return 1 - 2 * (inversions % 2)


def sign_permutation(config, perm_inv):
    """
    Fermionic sign for ONE configuration (no batch axis - vmap this over the
    MC-batch axis at the call site). Ns and n_bands are inferred from shapes:
        Ns      = perm_inv.shape[-1]
        n_bands = config.shape[-1] // Ns
    For each Ns-wide band slice, finds the occupied sites (ascending, padded
    to length Ns with sentinel Ns+1), maps them through `perm_inv`, and takes
    fermionic_sign; the total sign is the product over bands - generalizing
    a hardcoded up/down split to arbitrary Nbands.
    """
    Ns = perm_inv.shape[-1]
    n_bands = config.shape[-1] // Ns

    def _one_band(x_band):
        occ_idx = jnp.nonzero(x_band == 1, size=Ns, fill_value=Ns + 1)[0]
        mapped = jnp.take(perm_inv, occ_idx, fill_value=Ns + 1)
        return fermionic_sign(mapped)

    bands = config.reshape(n_bands, Ns)
    return jnp.prod(jax.vmap(_one_band)(bands))


def _combine_symmetrized_terms(outputs, sector_chars=None):
    """Shared combine step for symmetrize_wf: shifted, complex-safe
    log-sum-exp across the leading (group-element) axis.
    Generalizes singlet_symm's 2-term pattern to M = outputs.shape[0] terms.
    Not jax.scipy.special.logsumexp (never used anywhere in this codebase,
    since it needs jnp.max, undefined for complex log-amplitudes); shifting
    by the FIRST term rather than the max is this codebase's established
    idiom (see singlet_symm/time_reversal above).

    sector_chars, if given, is a real (M,) array of pi-multiples folded in as
    exp(i*pi*sector_chars) before combining - real +-1 characters only in
    this pass (general complex characters, e.g. momentum_phases, are a
    separate, harder follow-up: the "shift by first term" trick doesn't
    trivially generalize to complex weights).
    """
    if sector_chars is not None:
        outputs = outputs + 1j * jnp.pi * jnp.asarray(sector_chars)[:, None]
    zeroth = outputs[0]
    return zeroth + jnp.log(jnp.sum(jnp.exp(outputs - zeroth), axis=0))


def symmetrize_wf(wf_apply, perms, sector_chars=None):
    """Wrap a wavefunction to project onto a symmetric sector of a lattice
    permutation group (translations, point group, or any (M, W) perm array).

    Species-agnostic: goes through state.config/.replace_config, so the same
    wrapper works for SpinState and FermionState alike. If `state` is a
    FermionState, the fermionic-sign phase each group element picks up on
    the occupation-number representation is added automatically (via
    sign_permutation above), so the caller only ever supplies the single
    forward `perms` array - no separate inverse to build or pass in.

    The per-band inverse is derived from `perms` once here, at wrap time
    (argsort of the full (M, W) array), not on every call: band 0 of a
    properly expanded permutation (see expand_perm) only ever maps within
    [0, Ns), so argsort(perms)[..., :Ns] == argsort(perms[..., :Ns]) exactly
    - computing the full argsort up front and slicing to state.Ns per call
    is equivalent to, but cheaper than, re-deriving it from scratch inside
    the closure every forward pass.

    Unlike the bosonic case, a fermionic output is NOT literally constant
    across a group orbit: for a genuine sector eigenstate,
    f_sym(g.sigma) = f_sym(sigma) + 1j*pi*(sign(g,sigma) < 0) - log(chi(g))
    (trivial chi by default) - the same sign each individual term would pick
    up under g, not zero. This is expected, not a bug: a fermionic parity/
    momentum eigenstate genuinely transforms with a sign under the group, it
    isn't pointwise invariant.

    Args:
        wf_apply: Callable (params, state, *args, **kwargs) -> log-amplitude.
        perms: (M, W) int array, W == state.config.shape[-1] exactly. For a
            multi-band FermionState, build this with
            expand_perm(base_perms, state.Nbands).
        sector_chars: optional real (M,) array, see _combine_symmetrized_terms.

    Returns:
        Wrapped callable with the same signature returning the symmetrized
        log-amplitude.
    """
    perms = jnp.asarray(perms)
    # jnp.argsort (not the numpy-based invert_perm) computed once here, at
    # wrap time, on the full array; sliced to [..., :state.Ns] per call below.
    # numpy's generic argsort dispatch on a jax array breaks under jax.jit
    # tracing, so this must stay a jnp op even though it never sees a tracer
    # in practice (perms is concrete at wrap time).
    perms_inv_full = jnp.argsort(perms, axis=-1)

    def symmetrized_wf_apply(params, state, *args, **kwargs):
        def _apply_perm(perm_row):
            permuted = state.replace_config(state.config[..., perm_row])
            return wf_apply(params, permuted, *args, **kwargs)

        if isinstance(state, FermionState):
            perms_inv = perms_inv_full[..., :state.Ns]

            def _term(row):
                perm_row, perm_inv_row = row
                log_amp = _apply_perm(perm_row)
                sign = jax.vmap(sign_permutation, in_axes=(0, None))(state.config, perm_inv_row)
                return log_amp + 1j * jnp.pi * (sign < 0)

            outputs = jax.lax.map(jax.checkpoint(_term), (perms, perms_inv))
        else:
            outputs = jax.lax.map(jax.checkpoint(_apply_perm), perms)

        return _combine_symmetrized_terms(outputs, sector_chars)
    return symmetrized_wf_apply