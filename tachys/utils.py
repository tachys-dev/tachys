import jax
import jax.numpy as jnp
import numpy as np


def _cast_floating_to(tree, dtype):
    """Cast all floating-point leaves of a pytree to ``dtype``."""
    def conditional_cast(x):
        if isinstance(x, (np.ndarray, jnp.ndarray)) and jnp.issubdtype(x.dtype, jnp.floating):
            x = x.astype(dtype)
        return x
    return jax.tree_util.tree_map(conditional_cast, tree)
