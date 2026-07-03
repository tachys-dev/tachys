from flax import struct
import jax
import jax.numpy as jnp

from tachys.lattice.operator.base import _Operator, _OnSiteOperator, DiagonalResult, OffdiagonalResult


def _create_fermion_fn(state, site):
        size = state.occupations.shape[-1]

        new_occupations = state.occupations
        new_occupations = new_occupations.at[..., site].set(1)
        mask = (state.occupations[..., site] == 0)

        count = ((jnp.arange(size) < site) * state.occupations).sum(axis=-1)
        fermionic_sign = (-1)**count

        return state.replace(occupations=new_occupations), mask, fermionic_sign

def _destroy_fermion_fn(state, site):
        size = state.occupations.shape[-1]

        new_occupations = state.occupations
        new_occupations = new_occupations.at[..., site].set(0)
        mask = (state.occupations[..., site] == 1)

        count = ((jnp.arange(size) < site) * state.occupations).sum(axis=-1)
        fermionic_sign = (-1)**count

        return state.replace(occupations=new_occupations), mask, fermionic_sign

class C(_OnSiteOperator):
    band: int = struct.field(pytree_node=False)

    def apply(self, state):
        assert state.occupations.ndim == 2
        
        connected_states, mask, fermionic_sign = _destroy_fermion_fn(state, self.band * state.Ns + self.site)
        matrix_element = fermionic_sign * self.coupling

        return OffdiagonalResult(connected_states=connected_states,
                                 mask=mask,
                                 matrix_element=matrix_element)

class C_dag(_OnSiteOperator):
    band: int = struct.field(pytree_node=False)

    def apply(self, state):
        assert state.occupations.ndim == 2

        connected_states, mask, fermionic_sign = _create_fermion_fn(state, self.band * state.Ns + self.site)
        matrix_element = fermionic_sign * self.coupling

        return OffdiagonalResult(connected_states=connected_states,
                                 mask=mask,
                                 matrix_element=matrix_element)

class N(_OnSiteOperator):
    band: int = struct.field(pytree_node=False)

    def apply(self, state):
        assert state.occupations.ndim == 2
        occ = state.occupations[..., self.band * state.Ns + self.site]
        return DiagonalResult(matrix_element=occ * self.coupling)

class Nup(N):
    band: int = struct.field(pytree_node=False, default=0)

class Ndn(N):
    band: int = struct.field(pytree_node=False, default=1)

class Sz_f(_OnSiteOperator):
    """0.5*(n_up - n_dn): the z-spin of the fermionic (conduction-electron)
    degree of freedom at a site."""

    def apply(self, state):
        assert state.occupations.ndim == 2
        assert 2*state.Ns == state.occupations.shape[-1]
        n_up = state.occupations[..., self.site]
        n_dn = state.occupations[..., state.Ns + self.site]
        return DiagonalResult(matrix_element=0.5 * (n_up - n_dn) * self.coupling)

class Cup(C):
    band: int = struct.field(pytree_node=False, default=0)

class Cdn(C):
    band: int = struct.field(pytree_node=False, default=1)

class Cup_dag(C_dag):
    band: int = struct.field(pytree_node=False, default=0)

class Cdn_dag(C_dag):
    band: int = struct.field(pytree_node=False, default=1)


class Hopping(_Operator):
    '''alpha * c_dag(i) c(j) + conj(alpha) * c_dag(j) c(i).'''
    i: int
    j: int
    band: int = struct.field(pytree_node=False)

    def apply(self, state):
        assert state.occupations.ndim == 2
        occ = state.occupations
        size = occ.shape[-1]

        i = self.band * state.Ns + self.i
        j = self.band * state.Ns + self.j

        count1 = ((jnp.arange(size) < i) * occ).sum(axis=-1)
        count2 = ((jnp.arange(size) < j) * occ).sum(axis=-1)
        mask1 = i < j
        mask2 = i > j
        count = count1 + count2 - occ[..., i] * mask1 - occ[..., j] * mask2
        sign = (-1) ** (count % 2)

        new_occ = occ.at[..., i].set(occ[..., j])
        new_occ = new_occ.at[..., j].set(occ[..., i])
        mask = occ[..., i] != occ[..., j]
        connected_states = state.replace(occupations=new_occ)

        matrix_element = sign * jnp.where(occ[..., j] == 1, self.coupling, jnp.conj(self.coupling))

        return OffdiagonalResult(connected_states=connected_states,
                                 mask=mask,
                                 matrix_element=matrix_element)

class HoppingUp(Hopping):
    band: int = struct.field(pytree_node=False, default=0)

class HoppingDown(Hopping):
    band: int = struct.field(pytree_node=False, default=1)