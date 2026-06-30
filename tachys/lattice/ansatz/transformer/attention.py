from typing import Any
import jax
import jax.numpy as jnp

from flax import linen as nn
from einops import rearrange
from functools import partial

def roll(alpha, shift, axis=-1):
    return jnp.roll(alpha, shift, axis=axis)

@partial(jax.vmap, in_axes=(None, 0, None), out_axes=1)
@partial(jax.vmap, in_axes=(None, None, 0), out_axes=1)
def roll2d(weights, i, j):
    side_len = int(weights.shape[-1]**0.5)
    weights = weights.reshape(weights.shape[0], side_len, side_len)
    weights = jnp.roll(jnp.roll(weights, i, axis=-2), j, axis=-1)
    return weights.reshape(weights.shape[0], -1)

class FactoredAttention(nn.Module):
    """Flax multi-head attention with optional translational invariance.

    The attention weights alpha are shared across all query positions when
    `transl_invariant=True`: rather than a full (seq_len, seq_len) matrix per
    head, only a single row of length seq_len is learned and then shifted to
    cover every position via `jnp.roll`.

    When `two_dimensional=True` the sequence is interpreted as a flattened
    square lattice of side sqrt(seq_len), and translational invariance is
    enforced by rolling alpha in both spatial dimensions. This mode requires
    seq_len to be a perfect square.

    Args:
        d_model: Total embedding dimension (must be divisible by num_heads).
        num_heads: Number of attention heads.
        seq_len: Sequence length.
        dtype: Parameter dtype.
        transl_invariant: If True, enforce translational invariance via rolling.
        two_dimensional: If True, use 2D translational invariance. Only valid
            for square lattices (seq_len must be a perfect square). Requires
            transl_invariant=True.
    """

    d_model: int
    num_heads: int
    seq_len: int
    dtype: Any
    transl_invariant: bool = False
    two_dimensional: bool = False

    def setup(self):
        self.v = nn.Dense(self.d_model, kernel_init=nn.initializers.xavier_uniform(), param_dtype=self.dtype)
        if self.transl_invariant:
            self.alpha = self.param("alpha", nn.initializers.xavier_uniform(), (self.num_heads, self.seq_len), self.dtype)
            if self.two_dimensional:
                side_len = int(self.seq_len**0.5)
                assert side_len * side_len == self.seq_len
                self.alpha = roll2d(self.alpha, jnp.arange(side_len), jnp.arange(side_len))
                self.alpha = self.alpha.reshape(self.num_heads, -1, self.seq_len)
            else:
                self.alpha = jax.vmap(roll, (None, 0), out_axes=1)(self.alpha, jnp.arange(self.seq_len))
        else:
            self.alpha = self.param("alpha", nn.initializers.xavier_uniform(), (self.num_heads, self.seq_len, self.seq_len), self.dtype)

        self.W = nn.Dense(self.d_model, kernel_init=nn.initializers.xavier_uniform(), param_dtype=self.dtype)

    def __call__(self, x):
        v = self.v(x)
        v = rearrange(v, 'batch seq (num_heads head_dim) -> batch seq num_heads head_dim', num_heads=self.num_heads)
        v = rearrange(v, 'batch seq num_heads head_dim -> batch num_heads seq head_dim')
        x = jnp.matmul(self.alpha, v)
        x = rearrange(x, 'batch num_heads seq head_dim -> batch seq num_heads head_dim')
        x = rearrange(x, 'batch seq num_heads head_dim -> batch seq (num_heads head_dim)')

        x = self.W(x)

        return x