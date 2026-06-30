from typing import Any
from flax import linen as nn

from tachys.lattice.ansatz.transformer.attention import FactoredAttention


class EncoderBlock(nn.Module):
    d_model: int
    num_heads: int
    seq_len: int
    dtype: Any
    transl_invariant: bool = False
    two_dimensional: bool = False

    def setup(self):
        self.attn = FactoredAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            seq_len=self.seq_len,
            transl_invariant=self.transl_invariant,
            two_dimensional=self.two_dimensional,
            dtype=self.dtype,
        )
        self.layer_norm_1 = nn.LayerNorm(param_dtype=self.dtype)
        self.layer_norm_2 = nn.LayerNorm(param_dtype=self.dtype)
        self.ff = nn.Sequential([
            nn.Dense(4 * self.d_model, kernel_init=nn.initializers.xavier_uniform(), param_dtype=self.dtype),
            nn.gelu,
            nn.Dense(self.d_model, kernel_init=nn.initializers.xavier_uniform(), param_dtype=self.dtype),
        ])

    def __call__(self, x):
        x = x + self.attn(self.layer_norm_1(x))
        x = x + self.ff(self.layer_norm_2(x))
        return x


class Encoder(nn.Module):
    num_layers: int
    d_model: int
    num_heads: int
    seq_len: int
    dtype: Any
    transl_invariant: bool = False
    two_dimensional: bool = False

    def setup(self):
        self.layers = [
            EncoderBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                seq_len=self.seq_len,
                transl_invariant=self.transl_invariant,
                two_dimensional=self.two_dimensional,
                dtype=self.dtype,
            )
            for _ in range(self.num_layers)
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
