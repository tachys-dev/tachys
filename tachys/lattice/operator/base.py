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

class _Operator(struct.PyTreeNode):
    coupling: float = struct.field(default=1.0, kw_only=True)

    def __post_init__(self):
        import dataclasses
        for f in dataclasses.fields(self):
            if not f.metadata.get('pytree_node', True):
                continue
            val = getattr(self, f.name)
            if isinstance(val, jax.core.Tracer) or not isinstance(val, (int, float, jax.Array, np.ndarray, list, tuple)):
                continue
            if isinstance(val, tuple) and val and isinstance(val[0], _Operator):
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

    def apply(self, state):
        raise NotImplementedError()

    # ---- algebra ----
    def __add__(self, other):
        if type(self) is type(other):
            sd1 = fs.to_state_dict(self)
            sd2 = fs.to_state_dict(other)
            new_sd = jax.tree.map(
                lambda x, y: jnp.concatenate([jnp.atleast_1d(x), jnp.atleast_1d(y)], axis=0),
                sd1, sd2,
            )
            return fs.from_state_dict(self, new_sd)
        if isinstance(other, _Operator):
            return _OperatorSum(operators=(self, other))
        return NotImplemented

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
        if isinstance(other, _Operator):
            n_self  = jnp.atleast_1d(jnp.asarray(self.coupling)).shape[0]
            n_other = jnp.atleast_1d(jnp.asarray(other.coupling)).shape[0]
            if n_self != n_other:
                raise ValueError(
                    f"Cannot multiply operators with different coupling sizes: "
                    f"{type(self).__name__} has {n_self}, "
                    f"{type(other).__name__} has {n_other}"
                )
            return _OperatorMul(operators=(self, other))
        return NotImplemented
    __rmul__ = __mul__                          # scalar on the left

class _OperatorSum(_Operator):
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
        if isinstance(other, _Operator):
            for i, op in enumerate(self.operators):
                if type(op) is type(other):
                    merged = op + other
                    if type(merged) is type(op):  # actually merged, not wrapped in a new OperatorSum
                        new_ops = self.operators[:i] + (merged,) + self.operators[i+1:]
                        return _OperatorSum(operators=new_ops)
            return _OperatorSum(operators=self.operators + (other,))
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, _Operator):
            raise ValueError(
                "Cannot multiply a sum of operators; distribute the product manually "
                "(e.g. (A + B) * C → A*C + B*C)."
            )
        if isinstance(other, (int, float, complex)):
            # __call__ above never reads self.coupling, so storing the scalar
            # there would be silently dropped; distribute onto every term
            # instead (c * (A + B) == c*A + c*B, exact with no edge cases).
            return self.replace(operators=tuple(op * other for op in self.operators))
        return NotImplemented
    __rmul__ = __mul__                          # must be rebound here: _Operator.__rmul__
                                                 # is an alias to _Operator.__mul__ captured at
                                                 # class-definition time, so without this,
                                                 # `scalar * this_instance` would silently fall
                                                 # through to the base class's __mul__ instead
                                                 # of the override above.

class _OperatorMul(_Operator):
    operators: tuple

    def __call__(self, state):
        combined_mask = None
        combined_matrix_element = jnp.array([self.coupling])
        all_diagonal = True

        for op in reversed(self.operators):
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
        if isinstance(other, _Operator):
            return _OperatorSum(operators=(self, other))
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, _OperatorSum):
            raise ValueError(
                "Cannot multiply by a sum of operators; distribute the product manually "
                "(e.g. A * (B + C) → A*B + A*C)."
            )
        if isinstance(other, (int, float, complex)):
            # Push the scalar onto operators[0] rather than this wrapper's own
            # `coupling`: __add__'s same-treedef merge above rebuilds a fresh
            # _OperatorMul without preserving either side's outer coupling, so
            # anything stored there is silently lost the moment two
            # structurally-identical products are summed (the original bug).
            # operators[0] is always a genuine leaf-or-product factor, never
            # this same wrapper -- every code path that builds an _OperatorMul
            # appends new factors to the *end* of the tuple -- so this is exact
            # and terminates (recursing into this same branch again if
            # operators[0] happens to itself be an _OperatorMul). Scalar
            # placement doesn't matter mathematically: __call__ just
            # multiplies every factor's matrix element together.
            new_first = self.operators[0] * other
            return self.replace(operators=(new_first,) + self.operators[1:])
        if isinstance(other, _Operator):
            n_self  = jnp.atleast_1d(jnp.asarray(self.operators[0].coupling)).shape[0]
            n_other = jnp.atleast_1d(jnp.asarray(other.coupling)).shape[0]
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
