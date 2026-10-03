import numpy as np
import jax
import jax.numpy as jnp

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

mesh      = Mesh(jax.devices(), ('i',))
n_devices = len(jax.devices())
rank = jax.process_index()

# hard_shard and the NTK use rank as a device index: one device per process.
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
    lenght = array.shape[0]
    
    assert lenght % n_devices == 0

    lenght_per_proc = lenght // n_devices
    start, end = rank * lenght_per_proc, (rank+1) * lenght_per_proc
    return array[start:end]