"""Blurred sampling: an importance-weighted estimator on a smoothed density.

Z.-Q. Wan, R. Wiersema and S. Zhang, "Removing Nodal and Support-Mismatch
Pathologies in Variational Monte Carlo via Blurred Sampling", Phys. Rev. X 16,
031059 (2026). https://doi.org/10.1103/jrn5-gv19

The local estimator ``E_loc(x) = sum_x' <x|H|x'> psi(x')/psi(x)`` has heavy tails:
a walker that lands on a configuration with a very small ``|psi(x)|`` produces a
huge local energy, and because such configurations are rare under ``|psi|^2`` they
are sampled too seldom to average out. Blurred sampling smooths the sampling
density so that those configurations are visited at a controlled rate, and undoes
the resulting bias exactly with importance weights.

Concretely, once an ordinary Metropolis chain has produced ``x ~ p = |psi|^2/Z``,
each walker is moved with probability ``q`` to one of the ``N_conn(x)``
configurations connected to it by the off-diagonal part of the operator, chosen
uniformly. The batch is then distributed as

    p_blur(y) = (1-q) p(y) + q sum_x p(x) T(x->y),   T(x->y) = #{terms x->y} / N_conn(x)

and the exact reweighting factor back to ``p`` is (the normalisation ``Z`` cancels)

    w(y) = p(y) / p_blur(y) = 1 / [ (1-q) + q sum_{x~y} |psi(x)/psi(y)|^2 / N_conn(x) ]

The amplitude ratios in ``w`` are exactly the ones the local estimator already
computes, so ``blurred_local_estimator`` gets them from ``_offdiagonal_terms``
rather than re-evaluating the network. The only genuinely new cost is ``N_conn``
at every connected configuration, which is a mask evaluation with no network
calls (see ``n_conn_batch_size``).

Three things to keep in mind:

* **This is not a Markov chain move.** The blur is a post-processing of the
  sampled batch; the chain itself stays on ``|psi|^2`` and carries forward
  unblurred. ``tachys.montecarlo`` is not involved.
* **The connection multiset must be symmetric**, i.e. ``#{terms y->x} ==
  #{terms x->y}``, which holds for any Hermitian operator. The same operator has
  to drive both the proposal and the weight, or ``w`` is not the density ratio.
* **The blurred configurations must be in the ansatz's domain.** Walkers land on
  whatever the off-diagonal terms connect to, which is *not* restricted to the
  sector the Markov chain was confined to by its move. That is harmless when
  every off-diagonal term conserves what the ansatz assumes (a Heisenberg or
  Hubbard Hamiltonian preserves Sz and N, so a fixed-Sz or fixed-N determinant /
  Pfaffian head is safe), but a pairing or particle-non-conserving Hamiltonian
  will hand such a head a configuration outside its sector. Blur with an
  operator whose connectivity stays inside the ansatz's domain -- it need not be
  the full ``H``, only Hermitian and consistently used for both the proposal and
  the weight.

Typical use, through ``train``'s estimator protocol::

    from tachys.experimental.blurred_sampling import BlurredEstimator

    train(key, H, state, wf, optimizer, action, N_steps, lr_schedule, N_mc,
          estimator=BlurredEstimator(q=0.3))
"""

from functools import partial

import jax
import jax.numpy as jnp
from flax import struct
from jax import shard_map
from jax.sharding import PartitionSpec as P

from tachys.lattice.foundation.foundation_state import FoundationState
from tachys.lattice.operator.base import (
    DiagonalResult, DiagOffdiagResult, OffdiagonalResult,
)
from tachys.lattice.operator.local_estimator import _offdiagonal_terms, _operator_in_spec
from tachys.lattice.state_array import get_n_mc_local
from tachys.parallel import mesh, n_devices
from tachys.utils import _cast_floating_to


def _split_result(result):
    """(diagonal | None, offdiagonal) from any operator result.

    A purely diagonal operator connects no configurations, so there is nothing
    to blur with — that is a usage error, not an empty off-diagonal part.
    """
    if isinstance(result, DiagOffdiagResult):
        return result.diagonal, result.offdiagonal
    if isinstance(result, OffdiagonalResult):
        return None, result
    if isinstance(result, DiagonalResult):
        raise ValueError(
            "Blurred sampling needs an operator with off-diagonal terms, but got a "
            "purely diagonal one, which connects no configurations."
        )
    raise TypeError(f"Unexpected operator result type: {type(result)}")


def _n_conn(operator, state):
    """(N_mc,) number of active off-diagonal connections at each configuration."""
    _, offdiag = _split_result(operator(state))
    return offdiag.mask.astype(bool).sum(0)


# ─── Blur step ────────────────────────────────────────────────────────────────

def _blur_states(keys, operator, state, q):
    """Unsharded body of ``blur_states`` — see there for semantics."""
    N_mc_local = get_n_mc_local(state)

    keys = jax.vmap(jax.random.split)(keys)          # (N_mc_local, 2)
    key_pick, key_coin = keys[:, 0], keys[:, 1]

    _, offdiag = _split_result(operator(state))
    active   = offdiag.mask.astype(bool)             # (N_terms, N_mc_local)
    has_conn = active.any(axis=0)                    # (N_mc_local,)

    # jax.random.choice normalises p, so an all-zero column (a configuration with
    # no active off-diagonal connection: fully polarised, fully empty/full, ...)
    # would divide by zero and give a NaN index. Substitute a uniform p there and
    # force those walkers to stay put via `keep` below.
    p   = jnp.where(has_conn[None, :], active, True).astype(jnp.result_type(float)).T
    idx = jax.vmap(lambda p_i, k: jax.random.choice(k, p_i.size, p=p_i))(p, key_pick)

    moved = jax.tree.map(lambda x: x[idx, jnp.arange(N_mc_local)], offdiag.connected_states)

    keep = (jax.vmap(jax.random.uniform)(key_coin) > q) | ~has_conn

    def _select(original, new):
        mask = keep.reshape(keep.shape + (1,) * (original.ndim - 1))
        return jnp.where(mask, original, new)

    return jax.tree.map(_select, state, moved)


@jax.jit
def blur_states(keys, operator, state, q):
    """Draw a batch from the blurred density ``p_blur``.

    Each walker independently keeps its configuration with probability ``1-q``,
    and with probability ``q`` moves to one of the configurations connected to it
    by ``operator``'s off-diagonal part, chosen uniformly among the active ones.
    Walkers with no active connection always stay put.

    Parameters
    ----------
    keys     : jax.random.key array of shape (N_mc,), one key per chain — built
               the same way as ``sample``'s, i.e. ``jax.random.split(key, N_mc)``.
    operator : the operator whose off-diagonal connectivity defines the blur.
               Must be the same one used for the weights in
               ``compute_expectation_blurred``, and must be Hermitian.
    state    : State, batch axis 0 sharded across devices.
    q        : blur probability in [0, 1]. Traced, so annealing it costs no
               recompile. ``q = 0`` returns ``state`` unchanged.

    Returns
    -------
    state_blur : State with the same structure and batch size as ``state``.
    """
    return shard_map(
        _blur_states,
        mesh=mesh,
        in_specs=(P('i'), _operator_in_spec(operator), P('i'), P()),
        out_specs=P('i'),
        check_vma=False,
    )(keys, operator, state, q)


# ─── Weighted local estimator ─────────────────────────────────────────────────

def blurred_local_estimator(operator, state, wf, log_amps, q,
                            optimize_mask=True, batch_expand=1, n_conn_batch_size=None):
    """Local estimator and *unnormalised* blur weights on a blurred batch.

    Shares the whole off-diagonal pipeline with ``local_estimator``: one
    ``operator(state)`` call and one ``_offdiagonal_terms`` call give both the
    amplitude ratios that form ``O_L`` and the ones that form ``w``.

    Parameters
    ----------
    operator          : the same operator that drove ``blur_states``.
    state             : State (already blurred), batch axis 0 of size N_mc_local.
    wf                : WaveFunction.
    log_amps          : (N_mc_local,) log-amplitudes of ``state`` — of the *blurred*
                        configurations, not the ones the sampler returned.
    q                 : blur probability, matching ``blur_states``.
    optimize_mask,
    batch_expand      : forwarded to ``_offdiagonal_terms``.
    n_conn_batch_size : ``jax.lax.map`` batch size for the per-connection
                        ``N_conn`` sweep, which is what blurring actually costs.
                        ``None`` (default) maps one term at a time. XLA
                        dead-code-eliminates the connected states the sweep never
                        reads, so it is mask-only and adds almost nothing to peak
                        memory, but it is serial over the terms. Raising this
                        vectorises it, linearly in memory. Measured on a 6x6
                        Heisenberg model (144 off-diagonal terms, N_mc=256, CPU,
                        small RBM) against the unblurred ``compute_expectation``:

                            None -> 6.3x slower,   +1 MiB
                               4 -> 3.7x slower,   +8 MiB
                              16 -> 3.0x slower,  +31 MiB
                             144 -> 2.7x slower, +280 MiB

                        16 is a reasonable starting point. The ratio shrinks with
                        a more expensive ansatz, where the network evaluations in
                        the local estimator dominate instead.

    Returns
    -------
    O_L   : (N_mc_local,) local estimator
    w_raw : (N_mc_local,) blur weights, not yet normalised to mean 1
    """
    diag, offdiag = _split_result(operator(state))

    matrix_element, mask, psi_ratio = _offdiagonal_terms(
        offdiag, wf, log_amps, optimize_mask=optimize_mask, batch_expand=batch_expand,
    )

    O_L = jnp.sum(matrix_element * mask * psi_ratio, axis=0)
    if diag is not None:
        O_L = O_L + jnp.sum(diag.matrix_element, axis=0)

    # N_conn at each *connected* configuration, (N_terms, N_mc_local). Masks only,
    # so this never touches the network; it is the one cost blurring adds.
    n_conn = jax.lax.map(
        partial(_n_conn, operator), offdiag.connected_states, batch_size=n_conn_batch_size,
    )

    # `mask` is boolean in tachys (the fermionic sign lives in matrix_element), so
    # take |psi_ratio|^2 directly. The clamp only guards inactive entries, which
    # are zeroed anyway: if a term is active at y then the configuration it
    # connects to has at least the reverse connection, hence N_conn >= 1.
    active   = mask.astype(bool)
    contrib  = jnp.where(active, jnp.abs(psi_ratio) ** 2 / jnp.maximum(n_conn, 1), 0.0)
    w_raw    = 1.0 / (1.0 - q + q * jnp.sum(contrib, axis=0))

    # A configuration with no active connection is a fixed point of the blur: it
    # can never be left (blur_states keeps those walkers put) and, by symmetry of
    # the connection multiset, can never be reached either. So p_blur == p there
    # and its weight is exactly 1, not the 1/(1-q) the general formula gives.
    return O_L, jnp.where(active.any(axis=0), w_raw, 1.0)


def _compute_expectation_blurred(operator, wf, state, q,
                                 optimize_mask=True, batch_expand=1, n_conn_batch_size=None):
    """Unsharded body of ``compute_expectation_blurred`` — see there."""
    wf, state = _cast_floating_to((wf, state), wf.dtype)

    # The blurred configurations are new, so the sampler's log-amplitudes do not
    # apply: one extra forward pass over the batch.
    log_amps = wf.apply_fn(wf.params, state).astype(jnp.complex128)

    O_L, w = blurred_local_estimator(
        operator, state, wf, log_amps, q,
        optimize_mask=optimize_mask, batch_expand=batch_expand,
        n_conn_batch_size=n_conn_batch_size,
    )

    mean_w  = jax.lax.psum(jnp.mean(w),          'i') / n_devices
    mean_w2 = jax.lax.psum(jnp.mean(w ** 2),     'i') / n_devices
    ess     = mean_w ** 2 / mean_w2

    # E_{p_blur}[w] == 1 exactly, so this is self-normalisation: it turns the
    # weighted moments below into ratio estimators, which is what makes them
    # unbiased at finite sample size.
    w = w / mean_w

    O_mean  = jax.lax.psum(jnp.mean(w * O_L),                 'i') / n_devices
    O2_mean = jax.lax.psum(jnp.mean(w * jnp.abs(O_L) ** 2),   'i') / n_devices

    return O_L, w, O_mean, O2_mean, ess, mean_w


@partial(jax.jit, static_argnames=('optimize_mask', 'batch_expand', 'n_conn_batch_size'))
def compute_expectation_blurred(operator, wf, state, q,
                                optimize_mask=True, batch_expand=1, n_conn_batch_size=None):
    """Sharded, importance-weighted expectation value on a blurred batch.

    The weighted sibling of ``local_estimator.compute_expectation``: same sharding
    layout, but ``state`` is expected to come from ``blur_states`` and the returned
    moments are weighted averages over ``p_blur`` that estimate the ``|psi|^2``
    averages.

    Parameters
    ----------
    operator          : the same operator that drove ``blur_states``.
    state             : blurred State, batch axis 0 sharded across devices.
    q                 : blur probability, matching ``blur_states``.
    optimize_mask,
    batch_expand,
    n_conn_batch_size : see ``blurred_local_estimator``.

    Returns
    -------
    O_L     : (N_mc_local,) local estimator, sharded
    weights : (N_mc_local,) blur weights normalised to global mean 1, sharded
    O_mean  : scalar — weighted <O>
    O2_mean : scalar — weighted <|O|^2>
    ess     : scalar — effective sample size as a fraction of N_mc,
              ``<w>^2 / <w^2>``. Drops towards 0 as the reweighting degrades;
              a run where this falls much below ~0.5 wants a smaller ``q``.
    mean_w  : scalar — the pre-normalisation mean weight. Exactly 1 in
              expectation, so its deviation from 1 is a pure Monte Carlo
              diagnostic.
    """
    return shard_map(
        partial(_compute_expectation_blurred,
                optimize_mask=optimize_mask, batch_expand=batch_expand,
                n_conn_batch_size=n_conn_batch_size),
        mesh=mesh,
        in_specs=(_operator_in_spec(operator), P(), P('i'), P()),
        out_specs=(P('i'), P('i'), P(), P(), P(), P()),
        check_vma=False,
    )(operator, wf, state, q)


# ─── train() estimator ────────────────────────────────────────────────────────

class BlurredEstimator(struct.PyTreeNode):
    """``train(..., estimator=...)`` plug-in implementing blurred sampling.

    Attributes
    ----------
    q                 : blur probability. ``q = 0`` reproduces standard sampling
                        exactly (unit weights, unblurred batch). Since ``q`` is
                        traced, annealing it with ``estimator.replace(q=...)``
                        between steps triggers no recompile.
    optimize_mask,
    batch_expand,
    n_conn_batch_size : see ``blurred_local_estimator``.
    """

    q: float
    optimize_mask: bool = struct.field(pytree_node=False, default=True)
    batch_expand: int   = struct.field(pytree_node=False, default=1)
    n_conn_batch_size: int = struct.field(pytree_node=False, default=None)

    def __call__(self, keys, H, wf, state, log_amps):
        if isinstance(state, FoundationState):
            raise NotImplementedError(
                "Blurred sampling is not supported for FoundationState: the weighted "
                "SR path raises NotImplementedError for per-system centering "
                "(tachys/optimizer/_kernels.py, center_ntk and center_sr_solution)."
            )

        q = jnp.asarray(self.q, dtype=jnp.result_type(float))
        state_blur = blur_states(keys, H, state, q)

        O_L, weights, e_mean, e2_mean, ess, mean_w = compute_expectation_blurred(
            H, wf, state_blur, q,
            optimize_mask=self.optimize_mask, batch_expand=self.batch_expand,
            n_conn_batch_size=self.n_conn_batch_size,
        )

        metrics = {"ess": float(ess), "mean_weight": float(mean_w)}
        return state_blur, O_L, weights, e_mean, e2_mean, metrics
