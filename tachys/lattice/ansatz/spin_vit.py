from typing import Any
import jax
import jax.numpy as jnp

from flax import linen as nn
from einops import rearrange

from tachys.lattice.ansatz.transformer.encoder import Encoder
from tachys.lattice.ansatz.rbm import log_cosh


def extract_patches1d(x, b):
    return rearrange(x, 'batch (seq_len b) -> batch seq_len b', b=b)

def extract_patches2d(x, b):
    batch = x.shape[0]
    seq_len = int((x.shape[1] // b**2)**0.5)
    x = x.reshape(batch, seq_len, b, seq_len, b)
    x = x.transpose(0, 1, 3, 2, 4)
    x = x.reshape(batch, seq_len, seq_len, -1)
    x = x.reshape(batch, seq_len * seq_len, -1)
    return x

class Embed(nn.Module):
    d_model: int
    b: int
    dtype: Any
    two_dimensional: bool = False

    def setup(self):
        if self.two_dimensional:
            self.extract_patches = extract_patches2d
        else:
            self.extract_patches = extract_patches1d

        self.embed = nn.Dense(self.d_model, kernel_init=nn.initializers.xavier_uniform(), param_dtype=self.dtype)

    def __call__(self, x):
        x = self.extract_patches(x, self.b)
        x = self.embed(x)
        return x


class OutputHead(nn.Module):
    d_model: int
    dtype: Any
    complex: bool

    def setup(self):
        self.out_layer_norm = nn.LayerNorm(param_dtype=self.dtype)
        self.norm2 = nn.LayerNorm(use_scale=True, use_bias=True, param_dtype=self.dtype)
        self.output_layer0 = nn.Dense(self.d_model, param_dtype=self.dtype, kernel_init=nn.initializers.xavier_uniform(), bias_init=jax.nn.initializers.zeros)

        if self.complex:
            self.norm3 = nn.LayerNorm(use_scale=True, use_bias=True, param_dtype=self.dtype)
            self.output_layer1 = nn.Dense(self.d_model, param_dtype=self.dtype, kernel_init=nn.initializers.xavier_uniform(), bias_init=jax.nn.initializers.zeros)

    def __call__(self, y):
        z = self.out_layer_norm(y.sum(axis=1))
        out = self.norm2(self.output_layer0(z))
        if self.complex:
            out_complex = self.norm3(self.output_layer1(z))
            out = out + 1j * out_complex
        return jnp.sum(log_cosh(out), axis=-1)


class SpinViT(nn.Module):
    num_layers: int
    d_model: int
    num_heads: int
    seq_len: int
    b: int
    complex: bool = True
    transl_invariant: bool = False
    two_dimensional: bool = False
    dtype = jnp.float64

    def setup(self):
        self.patches_and_embed = Embed(
            self.d_model, self.b,
            two_dimensional=self.two_dimensional,
            dtype=self.dtype,
        )
        self.encoder = Encoder(
            num_layers=self.num_layers,
            d_model=self.d_model,
            num_heads=self.num_heads,
            seq_len=self.seq_len,
            transl_invariant=self.transl_invariant,
            two_dimensional=self.two_dimensional,
            dtype=self.dtype,
        )
        self.output = OutputHead(self.d_model, dtype=self.dtype, complex=self.complex)

    @nn.remat
    def __call__(self, lattice):
        s = lattice.spins
        
        x = self.patches_and_embed(s)
        
        y = self.encoder(x)
        
        out = self.output(y)

        return out.astype(jnp.complex128)