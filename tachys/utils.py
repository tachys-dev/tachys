import jax
import jax.numpy as jnp
import numpy as np


def same_treedef(tree1, tree2):
    """Identical pytree structure (types + nesting + static fields).

    Uses repr() comparison because JAX >=0.10 / Flax >=0.12 changed PyTreeDef.__eq__
    to ignore the registered node type (e.g. Splus == Sminus under __eq__).
    """
    return repr(jax.tree.structure(tree1)) == repr(jax.tree.structure(tree2))

def same_treedef_and_avals(tree1, tree2):
    """Identical structure AND matching leaf shape/dtype."""
    leaves1, td1 = jax.tree.flatten(tree1)
    leaves2, td2 = jax.tree.flatten(tree2)
    if repr(td1) != repr(td2):
        return False
    return all(
        jnp.shape(l1) == jnp.shape(l2) and jnp.result_type(l1) == jnp.result_type(l2)
        for l1, l2 in zip(leaves1, leaves2)
    )

def as_column(x):
    """1-d -> (N, 1); leaves other ranks unchanged."""
    x = jnp.asarray(x)
    return x[:, None] if x.ndim == 1 else x

def _cast_floating_to(tree, dtype):
    """Cast all floating-point leaves of a pytree to ``dtype``."""
    def conditional_cast(x):
        if isinstance(x, (np.ndarray, jnp.ndarray)) and jnp.issubdtype(x.dtype, jnp.floating):
            x = x.astype(dtype)
        return x
    return jax.tree_util.tree_map(conditional_cast, tree)
