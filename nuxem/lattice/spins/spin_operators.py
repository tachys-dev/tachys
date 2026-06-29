import jax
import jax.numpy as jnp

from ..operator.base import _Operator, _OnSiteOperator, DiagonalResult, OffdiagonalResult


class Sz(_OnSiteOperator):
    def apply(self, state):
        spins = state.spins
        assert spins.ndim == 2

        sz = 0.5 * spins[:, self.site]
        return DiagonalResult(matrix_element=sz*self.coupling)

class Splus(_OnSiteOperator):
    def apply(self, state):
        spins = state.spins
        assert spins.ndim == 2
        
        mask = spins[:, self.site] == -1

        connected_spins = spins.at[:, self.site].set(-spins[:, self.site])
        connected_states = state.replace(spins=connected_spins)

        matrix_element = jnp.full(mask.shape, self.coupling)

        return OffdiagonalResult(connected_states=connected_states,
                                 mask=mask,
                                 matrix_element=matrix_element)

class Sminus(_OnSiteOperator):
    def apply(self, state):
        spins = state.spins
        assert spins.ndim == 2

        mask = spins[:, self.site] == 1

        connected_spins = spins.at[..., self.site].set(-spins[..., self.site])
        connected_states = state.replace(spins=connected_spins)

        matrix_element = jnp.full(mask.shape, self.coupling)

        return OffdiagonalResult(connected_states=connected_states,
                                 mask=mask,
                                 matrix_element=matrix_element)


class XYExchange(_Operator):
    '''S+(i)S-(j) + S-(i)S+(j): flips both spins, non-zero only when i and j differ.'''
    i: int
    j: int

    def __post_init__(self):
        if isinstance(self.i, jax.core.Tracer) or type(self.i) is object:
            return
        i = jnp.atleast_1d(jnp.array(self.i))
        j = jnp.atleast_1d(jnp.array(self.j))
        n = i.shape[0]
        coupling = jnp.broadcast_to(jnp.atleast_1d(jnp.asarray(self.coupling)), (n,)).copy()
        object.__setattr__(self, 'i', i)
        object.__setattr__(self, 'j', j)
        object.__setattr__(self, 'coupling', coupling)

    def apply(self, state):
        spins = state.spins
        assert spins.ndim == 2

        mask = spins[:, self.i] != spins[:, self.j]

        connected_spins = spins.at[:, self.i].set(-spins[:, self.i])
        connected_spins = connected_spins.at[:, self.j].set(-spins[:, self.j])
        connected_states = state.replace(spins=connected_spins)

        matrix_element = jnp.full(mask.shape, self.coupling)

        return OffdiagonalResult(connected_states=connected_states,
                                 mask=mask,
                                 matrix_element=matrix_element)
