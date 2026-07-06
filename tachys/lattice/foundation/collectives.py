import jax
import jax.numpy as jnp

from functools import partial

def grouped_sum(x, y, K, axis=0):
    """Sums x into K groups given by y, reduced across the sharded mesh axis 'i'.

    Meant to be called inside a shard_map over mesh axis 'i': each shard
    computes a local per-group sum, then jax.lax.psum combines the shards
    so every shard ends up with the same, fully-summed (K, ...) result —
    groups whose elements are split across shards are still summed correctly.

    Args:
        x:    Local shard, any shape with x.shape[axis] == y.shape[0].
        y:    Local shard of integer group labels in [0, K).
        K:    Number of groups. Must be a static (non-traced) Python int,
              since it is used as segment_sum's num_segments.
        axis: Axis of x indexed by y; replaced by K in the output.

    Returns:
        Array like x but with size K along `axis`, replicated over the mesh.
    """
    xf    = jnp.moveaxis(x, axis, 0)                     # (n_local, ...)
    local = jax.ops.segment_sum(xf, y, num_segments=K)  # (K, ...)
    local = jnp.moveaxis(local, 0, axis)                # K restored to `axis`
    return jax.lax.psum(local, axis_name='i')           # replicated over mesh


def grouped_mean(x, y, K, axis=0, broadcast=False):
    """Averages x within each of K groups given by y.

    Same contract as grouped_sum, but averages within each group instead
    of summing. Counts are reduced the same way (per-shard segment_sum
    then psum) so groups split across shards are still averaged correctly.

    Args:
        x:         Local shard, any shape with x.shape[axis] == y.shape[0].
        y:         Local shard of integer group labels in [0, K).
        K:         Number of groups (static Python int).
        axis:      Axis of x indexed by y.
        broadcast: If False (default), return the reduced (..., K, ...)
                   array of per-group means, replicated over the mesh.
                   If True, return an array shaped like x instead, with
                   each element replaced by its own group's mean — e.g.
                   to center per-group values (such as per-system local
                   energies) via `x - grouped_mean(x, y, K, axis, True)`.
                   Note this changes the sharding of the result along
                   `axis` from replicated to matching x's local shard, so
                   the shard_map call site's out_specs must be updated
                   accordingly (e.g. P('i') instead of P()).

    Returns:
        Per-group means, shaped (..., K, ...) if broadcast=False, or
        shaped like x if broadcast=True.
    """
    total  = grouped_sum(x, y, K, axis=axis)
    counts = grouped_sum(jnp.ones_like(y, dtype=x.dtype), y, K, axis=0)  # (K,)

    shape = [1] * total.ndim
    shape[axis] = K
    mean = total / jnp.reshape(counts, shape)

    if broadcast:
        return jnp.take(mean, y, axis=axis)
    return mean

