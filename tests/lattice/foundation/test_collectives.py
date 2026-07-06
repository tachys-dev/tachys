"""Tests for tachys.lattice.foundation.collectives (grouped_sum, grouped_mean).

Both primitives are only meaningful inside a shard_map over the 'i' mesh
axis (they end with a jax.lax.psum(..., axis_name='i')), so every test
below drives them through shard_map + tachys.parallel.mesh rather than
calling them bare.

The in-process tests check correctness against a plain, unsharded
reference on whatever device configuration is available (this machine's
real device(s)). The *_multidevice tests additionally spawn a clean
subprocess with several virtual CPU devices forced via
XLA_FLAGS=--xla_force_host_platform_device_count, so the cross-shard
psum reduction is genuinely exercised even on a single-GPU/single-CPU
development machine.
"""
import json
import os
import subprocess
import sys
import tempfile
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

from tachys.parallel import mesh, n_devices
from tachys.lattice.foundation.collectives import grouped_sum, grouped_mean

_RUNNER = Path(__file__).parent / "run_collectives_multidevice.py"
_FORCED_DEVICE_COUNT = 4


def _reference_sum(x, y, K, axis):
    return jnp.moveaxis(
        jax.ops.segment_sum(jnp.moveaxis(x, axis, 0), y, num_segments=K), 0, axis)


def _reference_mean(x, y, K, axis):
    total = _reference_sum(x, y, K, axis)
    counts = jnp.bincount(y, length=K).astype(x.dtype)
    shape = [1] * total.ndim
    shape[axis] = K
    return total / counts.reshape(shape)


def _run(fn, x, y, K, axis):
    """Drive `fn` (grouped_sum or grouped_mean) through shard_map over the
    real mesh, sharding x/y along `axis`/0 exactly like main_foundation.py."""
    return shard_map(
        partial(fn, K=K, axis=axis),
        mesh=mesh,
        in_specs=(P(None, 'i'), P('i')),
        out_specs=P(),
    )(x, y)


# ---------------------------------------------------------------------------
# In-process correctness (runs on this machine's real device count)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("axis", [0, 1])
def test_grouped_sum_matches_reference(axis):
    K, n_features, n_per_device = 5, 3, 7
    n_elements = n_per_device * n_devices

    key = jax.random.key(0)
    kx, ky = jax.random.split(key)
    y = jax.random.permutation(ky, jnp.arange(n_elements) % K)

    if axis == 0:
        x = jax.random.normal(kx, (n_elements, n_features))
    else:
        x = jax.random.normal(kx, (n_features, n_elements))

    out = _run(grouped_sum, x, y, K, axis)
    ref = _reference_sum(x, y, K, axis)

    assert out.shape == ref.shape
    assert jnp.allclose(out, ref)


@pytest.mark.parametrize("axis", [0, 1])
def test_grouped_mean_matches_reference(axis):
    K, n_features, n_per_device = 5, 3, 7
    n_elements = n_per_device * n_devices

    key = jax.random.key(1)
    kx, ky = jax.random.split(key)
    y = jax.random.permutation(ky, jnp.arange(n_elements) % K)

    if axis == 0:
        x = jax.random.normal(kx, (n_elements, n_features))
    else:
        x = jax.random.normal(kx, (n_features, n_elements))

    out = _run(grouped_mean, x, y, K, axis)
    ref = _reference_mean(x, y, K, axis)

    assert out.shape == ref.shape
    assert jnp.allclose(out, ref)


def test_grouped_mean_equals_sum_over_counts():
    """grouped_mean should be exactly grouped_sum divided by per-group counts."""
    K, n_features, n_per_device = 4, 2, 6
    n_elements = n_per_device * n_devices
    axis = 1

    key = jax.random.key(2)
    kx, ky = jax.random.split(key)
    x = jax.random.normal(kx, (n_features, n_elements))
    y = jax.random.permutation(ky, jnp.arange(n_elements) % K)

    total = _run(grouped_sum, x, y, K, axis)
    mean = _run(grouped_mean, x, y, K, axis)
    counts = jnp.bincount(y, length=K).astype(x.dtype)

    assert jnp.allclose(mean, total / counts.reshape(1, K))


def test_grouped_sum_single_group_collects_everything():
    """K=1 puts every element in the same group: result is the full sum."""
    n_features, n_per_device = 3, 5
    n_elements = n_per_device * n_devices
    axis = 1

    x = jax.random.normal(jax.random.key(3), (n_features, n_elements))
    y = jnp.zeros(n_elements, dtype=jnp.int32)

    out = _run(grouped_sum, x, y, K=1, axis=axis)
    assert out.shape == (n_features, 1)
    assert jnp.allclose(out[:, 0], x.sum(axis=1))


# ---------------------------------------------------------------------------
# Genuine multi-device check: force several *distinct* CPU devices in a
# fresh subprocess (no mpirun required) so the psum actually combines
# partial sums computed on different physical devices.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def multidevice_results():
    fd, output_path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        env = os.environ.copy()
        env["JAX_PLATFORMS"] = "cpu"
        env["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={_FORCED_DEVICE_COUNT}"

        proc = subprocess.run(
            [sys.executable, str(_RUNNER), output_path],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode != 0:
            pytest.fail(
                f"runner exited with code {proc.returncode}\n"
                f"--- stdout ---\n{proc.stdout}\n"
                f"--- stderr ---\n{proc.stderr}"
            )
        with open(output_path) as f:
            return json.load(f)
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)


def test_multidevice_run_used_forced_device_count(multidevice_results):
    assert multidevice_results["n_devices"] == _FORCED_DEVICE_COUNT


def test_grouped_sum_matches_reference_multidevice(multidevice_results):
    sum_out = jnp.array(multidevice_results["sum_out"])
    ref_sum = jnp.array(multidevice_results["ref_sum"])
    assert jnp.allclose(sum_out, ref_sum)


def test_grouped_mean_matches_reference_multidevice(multidevice_results):
    mean_out = jnp.array(multidevice_results["mean_out"])
    ref_mean = jnp.array(multidevice_results["ref_mean"])
    assert jnp.allclose(mean_out, ref_mean)
