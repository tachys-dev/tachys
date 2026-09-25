from functools import partial
from typing import Any, Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
from flax import struct
from jax.sharding import PartitionSpec as P
from jax import shard_map

from tachys.parallel import mesh, n_devices, hard_shard
from tachys.optimizer._kernels import (
    _matmul_precision,
    linear_solver_cholesky,
    compute_ntk,
    center_sr_solution,
)
from tachys.lattice.foundation.foundation_state import FoundationState
from tachys.lattice.foundation.collectives import grouped_mean
from tachys.lattice.state_array import get_n_mc, get_n_mc_local
from tachys.utils import _cast_floating_to


# ─── Optimizer states ─────────────────────────────────────────────────────────

class SRState(NamedTuple):
    pass


class SPRINGState(NamedTuple):
    old_updates: Any  # pytree matching wf.params


class MARCHState(NamedTuple):
    old_updates: Any  # pytree matching wf.params
    V: Any            # exponential moving average of squared updates
    t: Any            # int32 step counter


# ─── Shared helper functions ───────────────────────────────────────────────────

def _make_apply_fn(raw_fn: Callable, mode: str) -> Callable:
    """Wrap the raw wavefunction to return real or (re, im) stacked output."""
    if mode == "complex":
        def apply_fn(params, state):
            log_amps = raw_fn(params, state)
            return jnp.concatenate((log_amps.real[:, None], log_amps.imag[:, None]), axis=-1)
        return apply_fn
    elif mode == "real":
        return lambda params, state: raw_fn(params, state).real
    else:
        raise ValueError(f"Unknown mode: {mode!r}")


def _eps(eloc: jax.Array, N_mc: int) -> jax.Array:
    """Force vector ε_i = 2 * conj(E_{Li} - Ē_L) / sqrt(M)."""
    return 2.0 * eloc.conj() / N_mc ** 0.5

def _center_eloc(E_L: jax.Array, state: Any, weights: Optional[jax.Array] = None) -> jax.Array:
    """Center local energies: per-system for foundation states, globally otherwise.

    On the reweighted path the weighted mean is used, so that `eloc` is the
    fluctuation about the energy estimator that is actually being minimized (and
    matches the weighted centering in `center_ntk` / `_jvp_correction`).
    `weights` are pre-normalized to mean 1 by `_call_reweighted`, so
    `mean(w * E_L)` already is the weighted mean.
    """
    # The constant subtracted here is arbitrary -- this centering could even be
    # dropped. center_ntk centers the Jacobian with the weighted mean, which puts
    # sqrt(w) in the null space of the centered O^T, so any offset in E_L cancels
    # out of the update: the plain mean would give the same answer here. Using
    # the weighted mean is a readability/conditioning choice, not a correctness
    # fix. See tests/optimizer/test_reweighted_sr.py.
    if isinstance(state, FoundationState):
        if weights is not None:
            raise NotImplementedError(
                "_center_eloc: weights + FoundationState is not supported yet."
            )
        return E_L - grouped_mean(E_L, state.system_ids, state.n_systems, broadcast=True)
    if weights is not None:
        return E_L - jax.lax.pmean(jnp.mean(weights * E_L), 'i')
    return E_L - jax.lax.pmean(jnp.mean(E_L), 'i')

def _jvp_correction(
    apply_fn: Callable,
    mode: str,
    params: Any,
    old_updates: Any,
    weights: Optional[jax.Array],
    N_mc: int,
    state: Any,
    dtype: Any = None,
) -> jax.Array:
    """JVP-based momentum correction for the gradient vector (SPRING / MARCH).

    Returns a local (hard-sharded) array of the same shape as eloc. With a
    ``dtype``, the JVP runs in it and its result is promoted back to at least
    float64 before the centering.
    """
    if dtype is not None:
        params, old_updates, state = _cast_floating_to((params, old_updates, state), dtype)
    with _matmul_precision(dtype):
        _, Jdtheta = jax.jvp(lambda p: apply_fn(p, state), (params,), (old_updates,))
    if dtype is not None:
        Jdtheta = Jdtheta.astype(jnp.promote_types(Jdtheta.dtype, jnp.float64))
    Jdtheta = jax.lax.all_gather(Jdtheta, 'i', tiled=True)

    if mode == "complex":
        # Convert [re, im] back to complex; minus sign from the conjugate gradient.
        Jdtheta = Jdtheta[:, 0] - 1j * Jdtheta[:, 1]

    if weights is not None:
        weights_all = jax.lax.all_gather(weights, 'i', tiled=True)
        Jdtheta = (Jdtheta - jnp.mean(weights_all * Jdtheta)) / N_mc ** 0.5
    else:
        Jdtheta = (Jdtheta - jnp.mean(Jdtheta)) / N_mc ** 0.5

    return hard_shard(Jdtheta)


def _build_ntk(
    state: Any,
    wf: Any,
    mode: str,
    weights: Optional[jax.Array],
    nbatches: int,
    N_mc_local: int,
    V: Optional[Any] = None,
    dtype: Any = None,
) -> jax.Array:
    """Natural tangent kernel matrix, normalized by total chain count.

    Pass V (bias-corrected second moment) to enable MARCH preconditioning.
    """
    ntk = compute_ntk(state, wf, mode, weights=weights, V=V, nbatches=nbatches, dtype=dtype)
    return ntk / (N_mc_local * n_devices)


def _parameter_updates(
    apply_fn: Callable,
    state: Any,
    wf: Any,
    mode: str,
    diag_shift: Any,
    eps: jax.Array,
    ntk: jax.Array,
    weights: Optional[jax.Array],
    solver: Callable = linear_solver_cholesky,
    dtype: Any = None,
) -> Any:
    """Solve the linear system and map the solution back to parameter space.

    Runs: linear solve → center → hard-shard → VJP → psum.
    Returns a pytree with the same structure and dtypes as wf.params.

    ``solver`` is called as ``solver(ntk, eps, diag_shift, mode)`` and must return
    the layout ``center_sr_solution`` expects: ``(M,)`` in mode="real", ``(2M,)``
    holding ``[u, v]`` in mode="complex". Defaults to the Cholesky solver; the
    real-time dynamics driver passes ``linear_solver_eigh`` instead (see
    tachys.dynamics.tdvp), which regularizes by discarding small eigenvalues.

    With a ``dtype``, only the VJP runs in it: the solution is cast to the
    dtype of the ansatz output for it, and the updates back to the parameters'.
    """
    sr_solution = solver(ntk, eps, diag_shift, mode)
    sr_solution = center_sr_solution(sr_solution, state, mode, weights)
    sr_solution = hard_shard(sr_solution)

    params = wf.params
    if dtype is not None:
        params, state = _cast_floating_to((params, state), dtype)
    with _matmul_precision(dtype):
        log_amps, vjp_fn = jax.vjp(lambda p: apply_fn(p, state), params)
        updates = vjp_fn(sr_solution.astype(log_amps.dtype))[0]
    updates = jax.tree.map(lambda u, p: u.astype(p.dtype), updates, wf.params)
    return jax.lax.psum(updates, 'i')


# ─── Base class ────────────────────────────────────────────────────────────────

class _BaseOptimizer(struct.PyTreeNode):
    """Fields shared by the SR-family optimizers and TDVP.

    diag_shift : Tikhonov shift of the NTK.
    mode       : "real" or "complex". Static field.
    nbatches   : number of sub-batches each device splits its samples into to
                 build the NTK, to bound the Jacobian's memory. Static field.
    dtype      : precision of the network evaluations inside the update -- the
                 per-sample Jacobians, their contraction into the NTK, the VJP
                 and the SPRING/MARCH JVP. The optimizer's counterpart of
                 ``WaveFunction.dtype``, which covers sampling and the local
                 estimators instead. Static field. ``None`` (default) evaluates
                 in the dtype the parameters are stored in. A dtype (e.g.
                 ``jnp.float32``, or ``"float32"`` straight from a config) casts
                 the parameters and the state to it for those evaluations, runs
                 their matmuls at full precision (JAX's default for float32 on
                 recent NVIDIA GPUs is TF32), and shifts the Jacobian before the
                 contraction (see ``ntk_parallel_fn``). The NTK and its solve
                 stay in float64, and the updates come back in the parameters'
                 own dtype. float32 rounding moves the NTK's eigenvalues by
                 about 1e-8 of the largest one, so a smaller ``diag_shift`` can
                 leave the shifted NTK indefinite. The Cholesky solve then fails
                 and, as for any failed solve, the SR step is silently zero
                 (SPRING and MARCH keep only their momentum term). TDVP's eigh
                 solver drops those directions instead.
    """
    diag_shift: float
    mode: str     = struct.field(pytree_node=False)
    nbatches: int = struct.field(pytree_node=False, default=1)
    dtype: Any    = struct.field(pytree_node=False, default=None)

    def __post_init__(self):
        import dataclasses
        if self.dtype is not None:
            # A numpy dtype rather than jnp.float32, a class: it prints in
            # train's setup summary, which skips callables.
            object.__setattr__(self, 'dtype', jnp.dtype(self.dtype))
        for f in dataclasses.fields(self):
            if not f.metadata.get('pytree_node', True):
                continue
            val = getattr(self, f.name)
            if isinstance(val, jax.core.Tracer) or not isinstance(val, (int, float, jax.Array)):
                continue
            object.__setattr__(self, f.name, jnp.atleast_1d(val))

    def update(self, O_L, opt_state, state, wf, weights=None):
        raise NotImplementedError

    @jax.jit
    @partial(shard_map, mesh=mesh,
             in_specs=(P(),     P(),     P('i'), P(),     P('i')),
             out_specs=P(), check_vma=False)
    def _call(self, opt_state, state, wf, E_L):
        return self.update(E_L, opt_state, state, wf)

    @jax.jit
    @partial(shard_map, mesh=mesh,
             in_specs=(P(),     P(),     P('i'), P(),     P('i'), P('i')),
             out_specs=P(), check_vma=False)
    def _call_reweighted(self, opt_state, state, wf, E_L, weights):
        weights = weights / jax.lax.pmean(jnp.mean(weights), 'i')
        return self.update(E_L, opt_state, state, wf, weights)

    def __call__(self, E_L, opt_state, state, wf, weights=None):
        if weights is None:
            return self._call(opt_state, state, wf, E_L)
        return self._call_reweighted(opt_state, state, wf, E_L, weights)


# ─── SR ───────────────────────────────────────────────────────────────────────

class SR(_BaseOptimizer):
    """Stochastic Reconfiguration (natural gradient via NTK)."""

    def init(self, params) -> SRState:
        return SRState()

    def update(self, E_L, opt_state, state, wf, weights=None):
        apply_fn   = _make_apply_fn(wf.apply_fn, self.mode)
        N_mc_local = get_n_mc_local(state)
        N_mc       = get_n_mc(state)

        eloc = _center_eloc(E_L, state, weights)
        eps = _eps(eloc, N_mc)
        if weights is not None:
            eps = jnp.sqrt(weights) * eps
        eps = jax.lax.all_gather(eps, 'i', tiled=True)

        ntk     = _build_ntk(state, wf, self.mode, weights, self.nbatches, N_mc_local, dtype=self.dtype)
        updates = _parameter_updates(apply_fn, state, wf, self.mode, self.diag_shift, eps, ntk, weights,
                                     dtype=self.dtype)
        return updates, SRState()


# ─── SPRING ───────────────────────────────────────────────────────────────────

class SPRING(_BaseOptimizer, kw_only=True):
    """SR with Projected Nesterov-style momentum (SPRING).

    Incorporates a JVP-based momentum correction into the gradient vector,
    then adds momentum to the final parameter updates.

    Reference
    ---------
    G. Goldshlager, N. Abrahamsen, L. Lin, "A Kaczmarz-inspired approach to
    accelerate the optimization of neural network wavefunctions", Journal of
    Computational Physics 516, 113351 (2024).
    https://doi.org/10.1016/j.jcp.2024.113351
    """

    mu: float = 0.9

    def init(self, params) -> SPRINGState:
        return SPRINGState(old_updates=jax.tree.map(jnp.zeros_like, params))

    def update(self, E_L, opt_state, state, wf, weights=None):
        apply_fn   = _make_apply_fn(wf.apply_fn, self.mode)
        N_mc_local = get_n_mc_local(state)
        N_mc       = get_n_mc(state)

        eloc = _center_eloc(E_L, state, weights)
        correction = _jvp_correction(
            apply_fn, self.mode, wf.params, opt_state.old_updates, weights, N_mc, state,
            dtype=self.dtype,
        )
        eps = _eps(eloc, N_mc) - self.mu * correction
        if weights is not None:
            eps = jnp.sqrt(weights) * eps
        eps = jax.lax.all_gather(eps, 'i', tiled=True)

        ntk          = _build_ntk(state, wf, self.mode, weights, self.nbatches, N_mc_local, dtype=self.dtype)
        base_updates = _parameter_updates(apply_fn, state, wf, self.mode, self.diag_shift, eps, ntk, weights,
                                          dtype=self.dtype)

        updates = jax.tree.map(lambda x, y: x + self.mu * y, base_updates, opt_state.old_updates)
        return updates, SPRINGState(old_updates=updates)


# ─── MARCH ────────────────────────────────────────────────────────────────────

class MARCH(_BaseOptimizer, kw_only=True):
    """SPRING with an adaptive second-moment preconditioner (MARCH).

    Maintains an exponential moving average V of squared update differences
    and uses its bias-corrected value to scale the NTK and the final updates.

    Reference
    ---------
    Y. Gu, W. Li, H. Lin, B. Zhan, R. Li, Y. Huang, D. He, Y. Wu, T. Xiang,
    M. Qin, L. Wang, D. Lv, "Solving the Hubbard model with neural quantum
    states", Nature Communications 17, 7838 (2026).
    https://doi.org/10.1038/s41467-026-74028-6
    """

    mu: float = 0.95
    beta: float = 0.995

    def init(self, params) -> MARCHState:
        return MARCHState(
            old_updates=jax.tree.map(jnp.zeros_like, params),
            V=jax.tree.map(jnp.ones_like, params),
            t=jnp.array([0], dtype=jnp.int32),
        )

    def update(self, E_L, opt_state, state, wf, weights=None):
        apply_fn   = _make_apply_fn(wf.apply_fn, self.mode)
        N_mc_local = get_n_mc_local(state)
        N_mc       = get_n_mc(state)

        eloc = _center_eloc(E_L, state, weights)

        # Bias-corrected V used for both NTK and update scaling.
        V_bc = jax.tree.map(
            lambda v: v / (1 - self.beta ** (opt_state.t + 1)), opt_state.V
        )

        correction = _jvp_correction(
            apply_fn, self.mode, wf.params, opt_state.old_updates, weights, N_mc, state,
            dtype=self.dtype,
        )
        eps = _eps(eloc, N_mc) - self.mu * correction
        if weights is not None:
            eps = jnp.sqrt(weights) * eps
        eps = jax.lax.all_gather(eps, 'i', tiled=True)

        ntk          = _build_ntk(state, wf, self.mode, weights, self.nbatches, N_mc_local, V=V_bc,
                                  dtype=self.dtype)
        base_updates = _parameter_updates(apply_fn, state, wf, self.mode, self.diag_shift, eps, ntk, weights,
                                          dtype=self.dtype)

        updates = jax.tree.map(
            lambda x, y, v: x / (jnp.sqrt(v) + 1e-8) + self.mu * y,
            base_updates, opt_state.old_updates, V_bc,
        )

        dtheta2 = jax.tree.map(lambda x, y: jnp.abs(x - y) ** 2, updates, opt_state.old_updates)
        new_V   = jax.tree.map(
            lambda v, d: self.beta * v + (1 - self.beta) * d, opt_state.V, dtheta2
        )
        return updates, MARCHState(old_updates=updates, V=new_V, t=opt_state.t + 1)
