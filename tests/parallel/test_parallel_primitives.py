"""Atomic unit tests for the tachys parallel infrastructure.

These tests run in a single process and verify that the core parallel
primitives (mesh, n_devices, rank, hard_shard, all_unshard) are
consistent with the underlying JAX device configuration.
"""
import jax
import jax.numpy as jnp
import pytest


# ---------------------------------------------------------------------------
# mesh / n_devices / rank
# ---------------------------------------------------------------------------

def test_n_devices_matches_jax_device_count():
    from tachys.parallel import n_devices
    assert n_devices == len(jax.devices())


def test_rank_matches_jax_process_index():
    from tachys.parallel import rank
    assert rank == jax.process_index()


def test_mesh_axis_name_is_i():
    from tachys.parallel import mesh
    assert "i" in mesh.axis_names


def test_mesh_size_matches_n_devices():
    from tachys.parallel import mesh, n_devices
    assert mesh.size == n_devices


# ---------------------------------------------------------------------------
# hard_shard
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("total,dtype", [
    (8,  jnp.float32),
    (16, jnp.float64),
    (16, jnp.complex128),
])
def test_hard_shard_shape(total, dtype):
    """hard_shard returns exactly 1/n_devices of the leading axis."""
    from tachys.optimizer._kernels import hard_shard
    from tachys.parallel import n_devices

    x = jnp.ones(total, dtype=dtype)
    chunk = hard_shard(x)
    assert chunk.shape[0] == total // n_devices


@pytest.mark.parametrize("total", [8, 16])
def test_hard_shard_correct_slice(total):
    """hard_shard returns this rank's contiguous slice of the global array."""
    from tachys.optimizer._kernels import hard_shard
    from tachys.parallel import n_devices, rank

    x = jnp.arange(total, dtype=jnp.float64)
    chunk_size = total // n_devices
    expected = x[rank * chunk_size : (rank + 1) * chunk_size]
    assert jnp.array_equal(hard_shard(x), expected)


def test_hard_shard_2d():
    """hard_shard slices the leading axis only; inner shape is preserved."""
    from tachys.optimizer._kernels import hard_shard
    from tachys.parallel import n_devices, rank

    total, cols = 16, 4
    x = jnp.arange(total * cols, dtype=jnp.float64).reshape(total, cols)
    chunk_size = total // n_devices
    expected = x[rank * chunk_size : (rank + 1) * chunk_size]
    result = hard_shard(x)
    assert result.shape == (chunk_size, cols)
    assert jnp.array_equal(result, expected)


# ---------------------------------------------------------------------------
# all_unshard (requires JIT context for the sharding constraint)
# ---------------------------------------------------------------------------

def test_all_unshard_preserves_values():
    """all_unshard inside jit returns the same numerical values."""
    from tachys.parallel import all_unshard

    @jax.jit
    def f(x):
        return all_unshard(x)

    x = jnp.arange(16, dtype=jnp.float64).reshape(4, 4)
    assert jnp.array_equal(f(x), x)
