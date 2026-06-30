import jax
import jax.numpy as jnp


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