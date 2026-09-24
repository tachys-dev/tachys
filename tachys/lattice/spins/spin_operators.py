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

        # Row x of S+: <x|S+|x'> = 1 for x' = x with spin `site` lowered, so
        # it is nonzero only where that spin is up in x.
        mask = spins[:, self.site] == 1

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

        # Row x of S-: <x|S-|x'> = 1 for x' = x with spin `site` raised, so
        # it is nonzero only where that spin is down in x.
        mask = spins[:, self.site] == -1

        connected_spins = spins.at[..., self.site].set(-spins[..., self.site])
        connected_states = state.replace(spins=connected_spins)

        matrix_element = jnp.full(mask.shape, self.coupling)

        return OffdiagonalResult(connected_states=connected_states,
                                 mask=mask,
                                 matrix_element=matrix_element)


class Sx(_OnSiteOperator):
    def apply(self, state):
        spins = state.spins
        assert spins.ndim == 2

        mask = jnp.ones(spins.shape[0], dtype=bool)
        connected_spins = spins.at[:, self.site].set(-spins[:, self.site])
        connected_states = state.replace(spins=connected_spins)
        matrix_element = jnp.full(mask.shape, self.coupling * 0.5)

        return OffdiagonalResult(connected_states=connected_states,
                                 mask=mask,
                                 matrix_element=matrix_element)


class Sy(_OnSiteOperator):
    def apply(self, state):
        spins = state.spins
        assert spins.ndim == 2

        mask = jnp.ones(spins.shape[0], dtype=bool)
        connected_spins = spins.at[:, self.site].set(-spins[:, self.site])
        connected_states = state.replace(spins=connected_spins)
        # Row x of Sy: <up|Sy|down> = -i/2, <down|Sy|up> = +i/2 (S+ = Sx+iSy,
        # S- = Sx-iSy => Sy = -i/2 (S+ - S-)); with spins[site] the spin of x,
        # both cases collapse to -1j * 0.5 * spins[site].
        matrix_element = -1j * 0.5 * spins[:, self.site] * self.coupling

        return OffdiagonalResult(connected_states=connected_states,
                                 mask=mask,
                                 matrix_element=matrix_element)


class XYExchange(_Operator):
    '''S+(i)S-(j) + S-(i)S+(j): flips both spins, non-zero only when i and j differ.'''
    i: int
    j: int

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
