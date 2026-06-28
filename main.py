from flax import struct
import flax.serialization as fs
import jax.numpy as jnp
import numpy as np
from deepvmc.operator.spins import Sz, Sminus, Splus
from deepvmc.state.lattice.spins import SpinState

operator = Sz(3) * Sminus(0) * Splus(1) + Sminus(2) * Sz(3)
# print(operator)

state = SpinState(spins=jnp.array([[1, -1, 1, -1], [1, 1, 1, 1], [-1, -1, 1, -1]], dtype=jnp.int8), Ns=4)

result = operator(state)

print(result)
# print(result["offdiagonal"]["new_state"].spins.shape)
# print(result["offdiagonal"]["new_state"].spins.shape)
# print(fs.to_state_dict(sz))