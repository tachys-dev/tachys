"""Combine per-system Hamiltonians into a single foundation-model operator.

A foundation model shares one wave function across a Monte Carlo batch that
mixes samples from several distinct systems (see FoundationState). The
Hamiltonian has to mix the same way: instead of one coupling per term
(shape (n_terms,), shared by every sample), each term needs a per-sample
coupling (shape (n_terms, N_mc)) that supplies the right system's value for
each column of the batch -- vmapping `_Operator.apply` over the term axis
then peels each leaf operator's `coupling` down to exactly (N_mc,), which is
already what every `apply` implementation expects to combine elementwise
with state-derived quantities.

`combine_systems` builds that from a list of per-system operators -- each
built from the same template (identical term structure) but with different
coupling values, exactly as produced by e.g.
``[hubbard_square_pbc(L, U=U) for U in Us]`` -- by broadcasting every leaf
operator's 1D coupling out to (n_terms, n_mc_per_system) and concatenating
those along the sample axis.
"""
import jax.numpy as jnp

from tachys.lattice.operator.base import _OperatorSum, _OperatorMul
from tachys.utils import same_treedef


def broadcast_coupling(operator, n_mc_per_system):
    """Broadcast every leaf operator's 1D coupling to 2D.

    (n_terms,) -> (n_terms, n_mc_per_system). Every sample drawn from
    `operator`'s system sees the same per-term coupling, so the new
    trailing axis is a plain repeat, not a fresh value per sample.
    Structural fields (site, i, j, ...) are left untouched.
    """
    if isinstance(operator, (_OperatorSum, _OperatorMul)):
        return operator.replace(
            operators=tuple(broadcast_coupling(op, n_mc_per_system) for op in operator.operators)
        )
    coupling = jnp.atleast_1d(jnp.asarray(operator.coupling))
    broadcasted = jnp.broadcast_to(coupling[:, None], (coupling.shape[0], n_mc_per_system))
    return operator.replace(coupling=broadcasted)


def concatenate_couplings(operators):
    """Concatenate same-structure operators' couplings along axis=1.

    Each of `operators` must already have 2D (n_terms, n_mc_per_system)
    couplings (see `broadcast_coupling`) and share identical tree structure
    (e.g. built from the same Hamiltonian template with different coupling
    values). Returns a single operator whose coupling has shape
    (n_terms, sum of n_mc_per_system) -- one column per Monte Carlo sample
    across all systems.
    """
    first, *rest = operators
    for other in rest:
        if not same_treedef(first, other):
            raise ValueError(
                "Cannot combine operators with different structure -- "
                "all systems must be built from the same Hamiltonian template."
            )
    return _concatenate_couplings(operators)


def _concatenate_couplings(operators):
    first = operators[0]
    if isinstance(first, (_OperatorSum, _OperatorMul)):
        return first.replace(operators=tuple(
            _concatenate_couplings([op.operators[i] for op in operators])
            for i in range(len(first.operators))
        ))
    coupling = jnp.concatenate([op.coupling for op in operators], axis=1)
    return first.replace(coupling=coupling)


def combine_systems(operators, n_mc_per_system):
    """Combine per-system operators into one foundation-model operator.

    See module docstring. `n_mc_per_system` is the number of Monte Carlo
    walkers dedicated to each system; the combined operator's coupling has
    shape (n_terms, len(operators) * n_mc_per_system), matching the leading
    batch dimension of the paired FoundationState.
    """
    broadcasted = [broadcast_coupling(op, n_mc_per_system) for op in operators]
    return concatenate_couplings(broadcasted)
