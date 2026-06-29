import jax.numpy as jnp

from .base import _OnSiteOperator, DiagonalResult, OffdiagonalResult


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
