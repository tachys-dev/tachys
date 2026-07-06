from functools import partial
from typing import Any, Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
from flax import struct
from jax.sharding import PartitionSpec as P
from jax import shard_map

from tachys.parallel import mesh, n_devices, hard_shard
from tachys.optimizer._kernels import (
    linear_solver_cholesky,
    compute_ntk,
    center_sr_solution,
)
from tachys.lattice.foundation.foundation_state import FoundationState
from tachys.lattice.foundation.collectives import grouped_mean


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

def _center_eloc(E_L: jax.Array, state: Any) -> jax.Array:
    """Center local energies: per-system for foundation states, globally otherwise."""
    if isinstance(state, FoundationState):
        return E_L - grouped_mean(E_L, state.system_ids, state.n_systems, broadcast=True)
    return E_L - jax.lax.pmean(jnp.mean(E_L), 'i')

def _jvp_correction(
    apply_fn: Callable,
    mode: str,
    params: Any,
    old_updates: Any,
    weights: Optional[jax.Array],
    N_mc: int,
    state: Any,
) -> jax.Array:
    """JVP-based momentum correction for the gradient vector (SPRING / MARCH).

    Returns a local (hard-sharded) array of the same shape as eloc.
    """
    _, Jdtheta = jax.jvp(lambda p: apply_fn(p, state), (params,), (old_updates,))
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
) -> jax.Array:
    """Natural tangent kernel matrix, normalized by total chain count.

    Pass V (bias-corrected second moment) to enable MARCH preconditioning.
    """
    ntk = compute_ntk(state, wf, mode, weights=weights, V=V, nbatches=nbatches)
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
) -> Any:
    """Solve the linear system and map the solution back to parameter space.

    Runs: cholesky solve → center → hard-shard → VJP → psum.
    Returns a pytree with the same structure as wf.params.
    """
    sr_solution = linear_solver_cholesky(ntk, eps, diag_shift, mode)
    sr_solution = center_sr_solution(sr_solution, state, mode, weights)
    sr_solution = hard_shard(sr_solution)

    _, vjp_fn = jax.vjp(lambda params: apply_fn(params, state), wf.params)
    updates = vjp_fn(sr_solution)[0]
    return jax.lax.psum(updates, 'i')


# ─── Base class ────────────────────────────────────────────────────────────────

class _BaseOptimizer(struct.PyTreeNode):
    diag_shift: float
    mode: str     = struct.field(pytree_node=False)
    nbatches: int = struct.field(pytree_node=False, default=1)

    def __post_init__(self):
        import dataclasses
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
        N_mc_local = state.config.shape[0]
        N_mc       = N_mc_local * n_devices

        eloc = _center_eloc(E_L, state)
        eps = _eps(eloc, N_mc)
        if weights is not None:
            eps = jnp.sqrt(weights) * eps
        eps = jax.lax.all_gather(eps, 'i', tiled=True)

        ntk     = _build_ntk(state, wf, self.mode, weights, self.nbatches, N_mc_local)
        updates = _parameter_updates(apply_fn, state, wf, self.mode, self.diag_shift, eps, ntk, weights)
        return updates, SRState()


# ─── SPRING ───────────────────────────────────────────────────────────────────

class SPRING(_BaseOptimizer, kw_only=True):
    """SR with Projected Nesterov-style momentum (SPRING).

    Incorporates a JVP-based momentum correction into the gradient vector,
    then adds momentum to the final parameter updates.
    """

    mu: float = 0.9

    def init(self, params) -> SPRINGState:
        return SPRINGState(old_updates=jax.tree.map(jnp.zeros_like, params))

    def update(self, E_L, opt_state, state, wf, weights=None):
        apply_fn   = _make_apply_fn(wf.apply_fn, self.mode)
        N_mc_local = state.config.shape[0]
        N_mc       = N_mc_local * n_devices

        eloc = _center_eloc(E_L, state)
        correction = _jvp_correction(
            apply_fn, self.mode, wf.params, opt_state.old_updates, weights, N_mc, state
        )
        eps = _eps(eloc, N_mc) - self.mu * correction
        if weights is not None:
            eps = jnp.sqrt(weights) * eps
        eps = jax.lax.all_gather(eps, 'i', tiled=True)

        ntk          = _build_ntk(state, wf, self.mode, weights, self.nbatches, N_mc_local)
        base_updates = _parameter_updates(apply_fn, state, wf, self.mode, self.diag_shift, eps, ntk, weights)

        updates = jax.tree.map(lambda x, y: x + self.mu * y, base_updates, opt_state.old_updates)
        return updates, SPRINGState(old_updates=updates)


# ─── MARCH ────────────────────────────────────────────────────────────────────

class MARCH(_BaseOptimizer, kw_only=True):
    """SPRING with an adaptive second-moment preconditioner (MARCH).

    Maintains an exponential moving average V of squared update differences
    and uses its bias-corrected value to scale the NTK and the final updates.
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
        N_mc_local = state.config.shape[0]
        N_mc       = N_mc_local * n_devices

        eloc = _center_eloc(E_L, state)

        # Bias-corrected V used for both NTK and update scaling.
        V_bc = jax.tree.map(
            lambda v: v / (1 - self.beta ** (opt_state.t + 1)), opt_state.V
        )

        correction = _jvp_correction(
            apply_fn, self.mode, wf.params, opt_state.old_updates, weights, N_mc, state
        )
        eps = _eps(eloc, N_mc) - self.mu * correction
        if weights is not None:
            eps = jnp.sqrt(weights) * eps
        eps = jax.lax.all_gather(eps, 'i', tiled=True)

        ntk          = _build_ntk(state, wf, self.mode, weights, self.nbatches, N_mc_local, V=V_bc)
        base_updates = _parameter_updates(apply_fn, state, wf, self.mode, self.diag_shift, eps, ntk, weights)

        updates = jax.tree.map(
            lambda x, y, v: x / (jnp.sqrt(v) + 1e-8) + self.mu * y,
            base_updates, opt_state.old_updates, V_bc,
        )

        dtheta2 = jax.tree.map(lambda x, y: jnp.abs(x - y) ** 2, updates, opt_state.old_updates)
        new_V   = jax.tree.map(
            lambda v, d: self.beta * v + (1 - self.beta) * d, opt_state.V, dtheta2
        )
        return updates, MARCHState(old_updates=updates, V=new_V, t=opt_state.t + 1)
