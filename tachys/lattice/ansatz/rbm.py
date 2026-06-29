from typing import Any

import jax.numpy as jnp
import flax.linen as nn


def log_cosh(x):
    """Numerically stable log(cosh(x))."""
    sgn_x = -2 * jnp.signbit(x.real) + 1
    x = x * sgn_x
    return x + jnp.log1p(jnp.exp(-2.0 * x)) - jnp.log(2.0)


class SpinRBM(nn.Module):
    num_hidden: int
    dtype: Any = jnp.float64

    def setup(self):
        self.linear = nn.Dense(
            features=self.num_hidden,
            use_bias=True,
            param_dtype=self.dtype,
        )

    def __call__(self, lattice):
        x = lattice.spins.astype(self.dtype)
        x = self.linear(x)
        x = log_cosh(x)
        return jnp.sum(x, axis=-1)
