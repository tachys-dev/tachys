"""Atomic unit tests for the tachys parallel infrastructure.

These tests run in a single process and verify that the core parallel
primitives (mesh, n_devices, rank, hard_shard, all_unshard) are
consistent with the underlying JAX device configuration.
"""
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import pytest
from jax import shard_map
from jax.sharding import PartitionSpec as P


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


def test_several_gpus_per_process_raise():
    """Importing tachys in a process that sees several GPUs raises."""
    code = (
        "import jax\n"
        "jax.default_backend = lambda: 'gpu'\n"
        "jax.local_device_count = lambda backend=None: 2\n"
        "import tachys\n"
    )
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    proc = subprocess.run([sys.executable, "-c", code], env=env,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode != 0
    assert "one process per GPU" in proc.stderr


# ---------------------------------------------------------------------------
# hard_shard (inside shard_map over 'i', like every caller)
# ---------------------------------------------------------------------------

def _hard_shard_on_mesh(x):
    """Every device's hard_shard chunk of the replicated x, in device order."""
    from tachys.parallel import mesh, hard_shard

    f = shard_map(hard_shard, mesh=mesh, in_specs=P(), out_specs=P('i'),
                  check_vma=False)
    return jax.jit(f)(x)


@pytest.mark.parametrize("total,dtype", [
    (8,  jnp.float32),
    (16, jnp.float64),
    (16, jnp.complex128),
])
def test_hard_shard_shape(total, dtype):
    """hard_shard returns exactly 1/n_devices of the leading axis."""
    from tachys.parallel import n_devices

    x = jnp.ones(total, dtype=dtype)
    chunk = _hard_shard_on_mesh(x).addressable_shards[0].data
    assert chunk.shape[0] == total // n_devices


@pytest.mark.parametrize("total", [8, 16])
def test_hard_shard_correct_slice(total):
    """Device d gets the d-th contiguous chunk, so the chunks rebuild the array."""
    x = jnp.arange(total, dtype=jnp.float64)
    assert jnp.array_equal(_hard_shard_on_mesh(x), x)


def test_hard_shard_2d():
    """hard_shard slices the leading axis only; inner shape is preserved."""
    from tachys.parallel import n_devices

    total, cols = 16, 4
    x = jnp.arange(total * cols, dtype=jnp.float64).reshape(total, cols)
    result = _hard_shard_on_mesh(x)
    assert result.addressable_shards[0].data.shape == (total // n_devices, cols)
    assert jnp.array_equal(result, x)


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
