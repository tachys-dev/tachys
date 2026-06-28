from flax import struct
import jax
import jax.numpy as jnp

import flax.serialization as fs

def same_treedef(op1, op2):
    """Identical operator tree structure (types + nesting + static fields)."""
    return jax.tree.structure(op1) == jax.tree.structure(op2)

def same_treedef_and_avals(op1, op2):
    """Identical structure AND matching leaf shape/dtype."""
    leaves1, td1 = jax.tree.flatten(op1)
    leaves2, td2 = jax.tree.flatten(op2)
    if td1 != td2:
        return False
    return all(
        jnp.shape(l1) == jnp.shape(l2) and jnp.result_type(l1) == jnp.result_type(l2)
        for l1, l2 in zip(leaves1, leaves2)
    )

def as_column(x):
    """1-d -> (N, 1); leaves other ranks unchanged."""
    x = jnp.asarray(x)
    return x[:, None] if x.ndim == 1 else x

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

    def __call__(self, state):
        self = jax.tree.map(jnp.atleast_1d, self)
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
        return super().__mul__(other)   # scalar multiplication is fine

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
            return self.replace(coupling=self.coupling * other)
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


class _OnSiteOperator(_Operator):
    site: int

    def __post_init__(self):
        # During jax.vmap, tree_unflatten is called with either object() sentinels
        # (structural consistency check) or abstract Tracers (actual tracing).
        # Neither can be used for shape arithmetic, so skip normalization in those cases.
        if isinstance(self.site, jax.core.Tracer) or type(self.site) is object:
            return
        site = jnp.atleast_1d(jnp.array(self.site))
        n = site.shape[0]
        coupling = jnp.broadcast_to(jnp.atleast_1d(jnp.asarray(self.coupling)), (n,)).copy()

        object.__setattr__(self, 'site', site)
        object.__setattr__(self, 'coupling', coupling)
