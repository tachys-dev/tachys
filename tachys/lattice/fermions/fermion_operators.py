from flax import struct
import jax
import jax.numpy as jnp

from tachys.lattice.operator.base import _OnSiteOperator, DiagonalResult, OffdiagonalResult


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

class Cup(C):
    band: int = struct.field(pytree_node=False, default=0)

class Cdn(C):
    band: int = struct.field(pytree_node=False, default=1)

class Cup_dag(C_dag):
    band: int = struct.field(pytree_node=False, default=0)

class Cdn_dag(C_dag):
    band: int = struct.field(pytree_node=False, default=1)