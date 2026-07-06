from typing import Any

import jax
import jax.numpy as jnp
import flax.linen as nn

from tachys.lattice.ansatz.fermionic_transformer import _log_det
from tachys.lattice.ansatz.rbm import log_cosh


class SpinFoundationRBM(nn.Module):
    """SpinRBM generalized to a foundation-model setting.

    Each sample's Hamiltonian couplings (lattice.system_couplings) are
    concatenated to the spin configuration before the RBM layer, so a single
    shared network can distinguish which system a sample was drawn from.
    """
    hidden_units: int
    dtype: Any = jnp.float64
    complex: bool = False

    def setup(self):
        self.linear = nn.Dense(
            features=self.hidden_units,
            use_bias=True,
            param_dtype=self.dtype,
        )
        if self.complex:
            self.imag_linear = nn.Dense(
                features=self.hidden_units,
                use_bias=True,
                param_dtype=self.dtype,
            )

    def __call__(self, lattice):
        x = jnp.concatenate((lattice.spins, lattice.system_couplings), axis=-1)
        x = self.linear(x)
        if self.complex:
            x = x + 1j * self.imag_linear(x)
        x = log_cosh(x.astype(complex))
        return jnp.sum(x, axis=-1)


class FermionFoundationRBM(nn.Module):
    """FermionRBM generalized to a foundation-model setting.

    Each sample's Hamiltonian couplings (lattice.system_couplings) are
    concatenated to the occupation configuration before the backflow
    network, so a single shared network can distinguish which system a
    sample was drawn from. The bare Slater determinant orbitals stay
    system-independent; only the backflow correction is conditioned on the
    couplings.
    """
    hidden_units: int

    @nn.compact
    def __call__(self, lattice):
        lattice = jax.tree.map(jnp.atleast_2d, lattice)
        occupations = lattice.occupations
        Ns = occupations.shape[-1]
        couplings = lattice.system_couplings

        #Bare Slater Determinant orbitals, shared across the batch
        orbitals = self.param('orbitals', nn.initializers.xavier_uniform(), (lattice.Ne, Ns,), jnp.float64)

        #Construct the Backflow correction, conditioned on the system couplings, one per sample
        backflow_input = jnp.concatenate((occupations, couplings), axis=-1)
        backflow = nn.Dense(self.hidden_units, param_dtype=jnp.float64)(backflow_input)
        backflow = jax.nn.tanh(backflow)
        backflow = nn.Dense(Ns * lattice.Ne, param_dtype=jnp.float64)(backflow)
        backflow = backflow.reshape(occupations.shape[0], lattice.Ne, Ns)
        orbitals = orbitals[None] + backflow

        #Find the positions of the occupied sites, per sample
        R = jax.vmap(lambda o: o.nonzero(size=lattice.Ne)[0])(occupations)
        A = jnp.take_along_axis(orbitals, R[:, None, :], axis=2)
        return _log_det(A)
