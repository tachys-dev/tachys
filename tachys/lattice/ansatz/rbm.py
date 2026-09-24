from typing import Any

import jax.numpy as jnp
import flax.linen as nn
import jax

from tachys.lattice.ansatz.fermionic_transformer import _log_det

def log_cosh(x):
    """Numerically stable log(cosh(x))."""
    sgn_x = -2 * jnp.signbit(x.real) + 1
    x = x * sgn_x
    return x + jnp.log1p(jnp.exp(-2.0 * x)) - jnp.log(2.0)


class SpinRBM(nn.Module):
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
        x = lattice.spins
        x = self.linear(x)
        if self.complex:
            x = x + 1j * self.imag_linear(x)
        x = log_cosh(x.astype(complex))
        return jnp.sum(x, axis=-1)

class FermionRBM(nn.Module):
    hidden_units: int

    @nn.compact
    def __call__(self, lattice):
        occupations = lattice.occupations
        Ns = occupations.shape[-1]

        #Bare Slater Determinant orbitals, shared across the batch
        orbitals = self.param('orbitals', nn.initializers.xavier_uniform(), (lattice.Ne, Ns,), jnp.float64)

        #Construct the Backflow correction, one per sample
        backflow = nn.Dense(self.hidden_units, param_dtype=jnp.float64)(occupations)
        backflow = jax.nn.tanh(backflow)
        backflow = nn.Dense(Ns * lattice.Ne, param_dtype=jnp.float64)(backflow)
        backflow = backflow.reshape(occupations.shape[0], lattice.Ne, Ns)
        orbitals = orbitals[None] + backflow

        #Find the positions of the occupied sites, per sample
        R = jax.vmap(lambda o: o.nonzero(size=lattice.Ne)[0])(occupations)
        A = jnp.take_along_axis(orbitals, R[:, None, :], axis=2)
        return _log_det(A)