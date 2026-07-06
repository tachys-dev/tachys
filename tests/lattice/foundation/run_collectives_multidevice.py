"""Runner script for the multi-device collectives regression test.

Computes grouped_sum / grouped_mean via shard_map across every locally
visible JAX device and writes the results (and the unsharded reference
values) as JSON to the path given as the first command-line argument.

Typical invocation from the project root, forcing 4 virtual CPU devices
in a single process (no MPI required):

    JAX_PLATFORMS=cpu XLA_FLAGS="--xla_force_host_platform_device_count=4" \\
        python tests/lattice/foundation/run_collectives_multidevice.py /tmp/out.json
"""
import json
import sys
from functools import partial
from pathlib import Path

# Ensure the project root is on sys.path regardless of working directory.
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

from tachys.parallel import mesh, n_devices
from tachys.lattice.foundation.collectives import grouped_sum, grouped_mean

K = 5
n_features = 3
n_per_device = 7
n_elements = n_per_device * n_devices
axis = 1

key = jax.random.key(0)
kx, ky = jax.random.split(key)
x = jax.random.normal(kx, (n_features, n_elements))
# Every group gets exactly n_elements // K (+ remainder) members, then
# shuffled so groups are scattered across shards rather than aligned
# with shard boundaries — this makes the test sensitive to a broken
# cross-shard reduction (e.g. a missing psum).
y = jax.random.permutation(ky, jnp.arange(n_elements) % K)


def run(fn):
    return shard_map(
        partial(fn, K=K, axis=axis),
        mesh=mesh,
        in_specs=(P(None, 'i'), P('i')),
        out_specs=P(),
    )(x, y)


sum_out = run(grouped_sum)
mean_out = run(grouped_mean)

ref_sum = jnp.moveaxis(
    jax.ops.segment_sum(jnp.moveaxis(x, axis, 0), y, num_segments=K), 0, axis)
counts = jnp.bincount(y, length=K).astype(x.dtype)
count_shape = [1] * ref_sum.ndim
count_shape[axis] = K
ref_mean = ref_sum / counts.reshape(count_shape)

output_path = sys.argv[1]
with open(output_path, "w") as f:
    json.dump({
        "n_devices": n_devices,
        "sum_out": sum_out.tolist(),
        "mean_out": mean_out.tolist(),
        "ref_sum": ref_sum.tolist(),
        "ref_mean": ref_mean.tolist(),
    }, f)
