from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from tachys.lattice.state_array import get_n_mc, get_n_mc_local
from tachys.optimizer._kernels import linear_solver_eigh
from tachys.optimizer.optimizers import (
    _BaseOptimizer,
    _build_ntk,
    _center_eloc,
    _make_apply_fn,
    _parameter_updates,
)


class TDVPState(NamedTuple):
    """Empty — the TDVP velocity depends only on the current parameters.

    Kept (rather than dropped) so the dynamics driver mirrors ``train``'s
    ``(key, state, wf, opt_state, history)`` contract and can round-trip through
    ``tachys.checkpoint`` unchanged.
    """
    pass


def _rt_eps(eloc: jax.Array, N_mc: int) -> jax.Array:
    """Real-time force vector eps_i = i * conj(E_Li - Ebar_L) / sqrt(M).

    The real-time counterpart of ``tachys.optimizer.optimizers._eps``; see this
    module's docstring for why the factor of 2 is absent and the ``1j`` present.
    Feeding this to ``_parameter_updates`` returns ``thetadot = S^-1 Im F``
    directly, as the physical velocity, sign included.
    """
    return 1j * eloc.conj() / N_mc ** 0.5


class TDVP(_BaseOptimizer, kw_only=True):
    """Real-time TDVP velocity, with the NTK inverted by spectral truncation.

    Used exactly like an optimizer -- ``dtheta_dt, opt_state = tdvp(E_L,
    opt_state, state, wf)`` -- but what it returns is the physical time
    derivative ``dtheta/dt``, not a descent direction: advance the parameters
    with ``theta + dt * dtheta_dt`` (which is what the integrators in
    ``tachys.dynamics.integrators`` do), never with ``apply_gradients``, whose
    ``p - eta * g`` convention would reverse the direction of time.

    Fields
    ------
    diag_shift : Tikhonov shift applied to the *kept* eigenvalues,
                 ``1 / (lambda + diag_shift)``. Defaults to 0.0: with the
                 spectral truncation below, a shift is no longer needed to make
                 the solve well-posed, and it biases the directions that survive.
    mode       : must be ``"complex"``. Static field.
    nbatches   : NTK sub-batching, as for the SR-family optimizers. Static field.
    rcond      : relative eigenvalue cutoff -- eigenvalues at or below
                 ``rcond * lambda_max`` are discarded, capping the condition
                 number of the retained subspace at ``1 / rcond``. The single
                 most important knob of a t-VMC run; worth scanning, and worth
                 logging alongside the TDVP error (which is precisely the
                 diagnostic that prices it).
    atol       : absolute floor on that cutoff. Default 0.0 (purely relative).
    """

    diag_shift: float = 0.0
    rcond: float = 1e-8
    atol: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        if self.mode != "complex":
            raise ValueError(
                f"TDVP requires mode='complex', got mode={self.mode!r}. Real-time "
                "evolution generates a phase, and a real-valued log-amplitude has "
                "no parameter that can carry it -- the tangent space cannot "
                "represent -i H|psi>, so the resulting trajectory would be "
                "meaningless rather than merely inaccurate. Use a complex ansatz "
                "(e.g. SpinRBM(..., complex=True))."
            )

    def init(self, params) -> TDVPState:
        return TDVPState()

    def update(self, E_L, opt_state, state, wf, weights=None):
        apply_fn   = _make_apply_fn(wf.apply_fn, self.mode)
        N_mc_local = get_n_mc_local(state)
        N_mc       = get_n_mc(state)

        eloc = _center_eloc(E_L, state)
        eps  = _rt_eps(eloc, N_mc)
        if weights is not None:
            eps = jnp.sqrt(weights) * eps
        eps = jax.lax.all_gather(eps, 'i', tiled=True)

        ntk = _build_ntk(state, wf, self.mode, weights, self.nbatches, N_mc_local)
        dtheta_dt = _parameter_updates(
            apply_fn, state, wf, self.mode, self.diag_shift, eps, ntk, weights,
            solver=partial(linear_solver_eigh, rcond=self.rcond, atol=self.atol),
        )
        return dtheta_dt, TDVPState()
