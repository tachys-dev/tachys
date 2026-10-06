import numpy as np
import jax
import jax.numpy as jnp

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

mesh      = Mesh(jax.devices(), ('i',))
n_devices = len(jax.devices())
rank = jax.process_index()

# One GPU per process is the only multi-GPU layout that has been run.
# CPU is exempt, for the tests that force several CPU devices into one process.
if jax.default_backend() != "cpu" and jax.local_device_count() > 1:
    _n = jax.local_device_count()
    raise RuntimeError(
        f"tachys needs one process per GPU, but this process has {_n} GPUs: start "
        f"{_n} processes (srun --ntasks-per-node={_n}, mpirun -np {_n}), or make a "
        f"single GPU visible with CUDA_VISIBLE_DEVICES."
    )

MASTER = 0

def all_unshard(pytree):
    return jax.lax.with_sharding_constraint(pytree, NamedSharding(mesh, P()))


def promote_to_pytree(f):
    def f_pytree(pytree):
        return jax.tree.map(f, pytree)

    return f_pytree

@promote_to_pytree
def hard_shard(array):
    """This device's contiguous chunk of the leading axis; call inside shard_map over 'i'.

    The offset is axis_index('i'), not the Python int rank, so every process compiles
    the same program (XLA's multi-process GPU autotuning hangs otherwise).
    """
    lenght = array.shape[0]

    assert lenght % n_devices == 0

    lenght_per_proc = lenght // n_devices
    start = jax.lax.axis_index('i') * lenght_per_proc
    return jax.lax.dynamic_slice_in_dim(array, start, lenght_per_proc)