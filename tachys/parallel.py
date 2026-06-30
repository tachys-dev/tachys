import numpy as np
import jax
import jax.numpy as jnp

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

mesh      = Mesh(jax.devices(), ('i',))
n_devices = len(jax.devices())
rank = jax.process_index()

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