from flax import struct
import jax
import jax.numpy as jnp

import flax.serialization as fs
import numpy as np

from tachys.utils import same_treedef

class DiagonalResult(struct.PyTreeNode):
    matrix_element: jax.Array

class OffdiagonalResult(struct.PyTreeNode):
    connected_states: jax.Array
    mask: jax.Array
    matrix_element: jax.Array

class DiagOffdiagResult(struct.PyTreeNode):
    diagonal: DiagonalResult
    offdiagonal: OffdiagonalResult

class _OperatorBase(struct.PyTreeNode):
    """Common pytree base for operator-algebra nodes: leaf operators, sums,
    and composite products. Lets the algebra below treat "any operator-like
    node" uniformly via isinstance, without forcing composite (Sum/Mul)
    nodes to carry the leaf-only `coupling` field.
    """

    def _coupling_size(self):
        """Batched-term count for the coupling-size compatibility check in
        __mul__. Composites are never themselves batched, so 1 is exact --
        it matches what a composite's own `coupling` was always hard-coded
        to back when Sum/Mul still inherited that field.
        """
        return 1

    def apply(self, state):
        raise NotImplementedError()

    def __add__(self, other):
        if isinstance(other, _OperatorBase):
            return _OperatorSum(operators=(self, other))
        return NotImplemented

class _Operator(_OperatorBase):
    """Leaf operator O. ``apply(state)`` returns, for every configuration x of
    the batch, the row of O at x: the configurations x' with <x|O|x'> != 0 and
    those matrix elements. The local estimator
    sum_x' <x|O|x'> psi(x') / psi(x) then averages to <psi|O|psi> over
    |psi|^2, for any O, Hermitian or not. The full conventions -- basis
    ordering, wavefunction, estimators -- are stated in
    ``tachys.lattice.operator.local_estimator``.
    """
    coupling: float = struct.field(default=1.0, kw_only=True)

    def _coupling_size(self):
        return jnp.atleast_1d(jnp.asarray(self.coupling)).shape[0]

    def __post_init__(self):
        import dataclasses
        for f in dataclasses.fields(self):
            if not f.metadata.get('pytree_node', True):
                continue
            val = getattr(self, f.name)
            if isinstance(val, jax.core.Tracer) or not isinstance(val, (int, float, complex, np.generic, jax.Array, np.ndarray, list, tuple)):
                continue
            if isinstance(val, tuple) and val and isinstance(val[0], _OperatorBase):
                continue
            object.__setattr__(self, f.name, jnp.atleast_1d(jnp.asarray(val)))

    def __call__(self, state):
        # self = jax.tree.map(jnp.atleast_1d, self) # unnecessary
        leaves, treedef = jax.tree_util.tree_flatten(self)
        n = max((l.shape[0] for l in leaves if l.ndim >= 1), default=1)
        if n > 1:
            leaves = [
                jnp.broadcast_to(l, (n,) + l.shape[1:])
                if l.ndim >= 1 and l.shape[0] == 1 else l
                for l in leaves
            ]
            self = treedef.unflatten(leaves)

        #* type(self).apply is the unbound method
        result = jax.vmap(type(self).apply, in_axes=(0, None))(self, state)
        return result

    # ---- algebra ----
    def __add__(self, other):
        # same_treedef, not just type equality: the state-dict round-trip below
        # carries only pytree_node=True fields, so merging two instances that
        # differ in a static field (e.g. Hopping band=0 vs band=2) would keep
        # only self's value and silently produce a wrong operator. Mismatched
        # statics fall through to _OperatorSum, which __add__ there already
        # buckets by matching term. Mirrors _OperatorMul.__add__'s guard.
        if type(self) is type(other) and same_treedef(self, other):
            sd1 = fs.to_state_dict(self)
            sd2 = fs.to_state_dict(other)
            new_sd = jax.tree.map(
                lambda x, y: jnp.concatenate([jnp.atleast_1d(x), jnp.atleast_1d(y)], axis=0),
                sd1, sd2,
            )
            return fs.from_state_dict(self, new_sd)
        return super().__add__(other)

    def __sub__(self, other): raise NotImplementedError()
    def __neg__(self):        raise NotImplementedError()
    def __mul__(self, other):
        if isinstance(other, _OperatorSum):
            raise ValueError(
                "Cannot multiply by a sum of operators; distribute the product manually "
                "(e.g. A * (B + C) → A*B + A*C)."
            )
        if isinstance(other, (int, float, complex)):
            return self.replace(coupling=self.coupling * other)
        if isinstance(other, _OperatorBase):
            n_self  = self._coupling_size()
            n_other = other._coupling_size()
            if n_self != n_other:
                raise ValueError(
                    f"Cannot multiply operators with different coupling sizes: "
                    f"{type(self).__name__} has {n_self}, "
                    f"{type(other).__name__} has {n_other}"
                )
            return _OperatorMul(operators=(self, other))
        return NotImplemented
    __rmul__ = __mul__                          # scalar on the left

class _OperatorSum(_OperatorBase):
    operators: tuple

    def __call__(self, state):
        results = [op(state) for op in self.operators]

        diagonal_parts = [r for r in results if isinstance(r, DiagonalResult)]
        offdiagonal_parts = [r for r in results if isinstance(r, OffdiagonalResult)]

        if not offdiagonal_parts:
            return DiagonalResult(
                matrix_element=jnp.concatenate([r.matrix_element for r in diagonal_parts], axis=0)
            )

        merged_offdiag = jax.tree.map(
            lambda *xs: jnp.concatenate(xs, axis=0),
            *offdiagonal_parts,
        )

        if not diagonal_parts:
            return merged_offdiag

        merged_diag = DiagonalResult(
            matrix_element=jnp.concatenate([r.matrix_element for r in diagonal_parts], axis=0)
        )
        return DiagOffdiagResult(diagonal=merged_diag, offdiagonal=merged_offdiag)

    def __add__(self, other):
        if isinstance(other, _OperatorSum):
            result = self
            for op in other.operators:
                result = result + op
            return result
        if isinstance(other, _OperatorBase):
            for i, op in enumerate(self.operators):
                if type(op) is type(other):
                    merged = op + other
                    if type(merged) is type(op):  # actually merged, not wrapped in a new OperatorSum
                        new_ops = self.operators[:i] + (merged,) + self.operators[i+1:]
                        return _OperatorSum(operators=new_ops)
            return _OperatorSum(operators=self.operators + (other,))
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, _OperatorBase):
            raise ValueError(
                "Cannot multiply a sum of operators; distribute the product manually "
                "(e.g. (A + B) * C → A*C + B*C)."
            )
        if isinstance(other, (int, float, complex)):
            # _OperatorSum has no coupling field of its own to store the
            # scalar in; distribute onto every term instead
            # (c * (A + B) == c*A + c*B, exact with no edge cases).
            return self.replace(operators=tuple(op * other for op in self.operators))
        return NotImplemented
    __rmul__ = __mul__                          # must be rebound here: _OperatorBase defines
                                                 # neither __mul__ nor __rmul__, so without this,
                                                 # `scalar * this_instance` would raise TypeError
                                                 # instead of using the override above.

class _OperatorMul(_OperatorBase):
    operators: tuple

    def __call__(self, state):
        combined_mask = None
        combined_matrix_element = jnp.array([1.0]) # brodcasting will happen
        all_diagonal = True

        # Rows compose left to right, <x|A B|x''> = sum_x' <x|A|x'> <x'|B|x''>:
        # take the row of the leftmost factor at x, then the row of the next
        # factor at the configuration x' it connects to, and so on.
        for op in self.operators:
            if all_diagonal: #* this takes into account also the first offdiagonal application
                result = op(state)
            else:
                result = jax.vmap(type(op).apply, in_axes=(0, 0))(op, state)

            if isinstance(result, DiagonalResult):
                combined_matrix_element = combined_matrix_element * result.matrix_element
            else:
                all_diagonal = False
                state = result.connected_states
                combined_mask = (
                    result.mask if combined_mask is None
                    else (combined_mask & result.mask)
                )
                combined_matrix_element = combined_matrix_element * result.matrix_element

        if all_diagonal:
            return DiagonalResult(matrix_element=combined_matrix_element)

        return OffdiagonalResult(connected_states=state, mask=combined_mask, matrix_element=combined_matrix_element)

    def __add__(self, other):
        if isinstance(other, _OperatorMul) and same_treedef(self, other):
            return _OperatorMul(operators=tuple(x + y for x, y in zip(self.operators, other.operators)))
        return super().__add__(other)

    def __mul__(self, other):
        if isinstance(other, _OperatorSum):
            raise ValueError(
                "Cannot multiply by a sum of operators; distribute the product manually "
                "(e.g. A * (B + C) → A*B + A*C)."
            )
        if isinstance(other, (int, float, complex)):
            # Push the scalar onto operators[0] rather than a `coupling` field
            # of our own -- _OperatorMul has none (see _OperatorBase). This
            # also sidesteps __add__'s same-treedef merge above, which
            # rebuilds a fresh _OperatorMul without preserving anything from
            # the outer wrapper.
            # operators[0] is always a genuine leaf-or-product factor, never
            # this same wrapper -- every code path that builds an _OperatorMul
            # appends new factors to the *end* of the tuple -- so this is exact
            # and terminates (recursing into this same branch again if
            # operators[0] happens to itself be an _OperatorMul). Scalar
            # placement doesn't matter mathematically: __call__ just
            # multiplies every factor's matrix element together.
            new_first = self.operators[0] * other
            return self.replace(operators=(new_first,) + self.operators[1:])
        if isinstance(other, _OperatorBase):
            n_self  = self.operators[0]._coupling_size()
            n_other = other._coupling_size()
            if n_self != n_other:
                raise ValueError(
                    f"Cannot multiply operators with different coupling sizes: "
                    f"OperatorMul has {n_self}, "
                    f"{type(other).__name__} has {n_other}"
                )
            return _OperatorMul(operators=self.operators + (other,))
        return NotImplemented
    __rmul__ = __mul__                          # see _OperatorSum.__rmul__ for why this
                                                 # rebinding is required, not inherited.


class _OnSiteOperator(_Operator):
    site: int
