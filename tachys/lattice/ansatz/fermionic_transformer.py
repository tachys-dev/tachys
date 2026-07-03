import jax
import jax.numpy as jnp
from flax import linen as nn

from einops import einsum

from typing import Any

from tachys.lattice.ansatz.transformer.encoder import Encoder

def _log_det(A):
    sign, logabsdet = jnp.linalg.slogdet(A)
    cdtype = jnp.promote_types(A.dtype, jnp.complex64)
    amp = logabsdet.astype(cdtype) + jnp.log(sign.astype(cdtype))
    return jnp.where(jnp.isnan(amp), -jnp.inf, amp)

def compute_orbitals_fn(y, weights):
    return einsum(y, weights, 'batch Norb d, Norb d Ne-> batch Norb Ne')

class OutputHeadDet(nn.Module):
    d_model: int
    Ne: int
    Ns: int
    dtype: Any
    Nbands: int = 2

    def setup(self):
        self.W = self.param('W', nn.initializers.xavier_uniform(), (2*self.Ns, self.d_model, self.Ne), self.dtype)

    def __call__(self, y, R):
        #* y.shape = [batch, (Nbands*)Ns, d_model]
        y = jnp.concatenate(2*(y,), axis=1)

        orbitals = compute_orbitals_fn(y, self.W) #* shape = [batch, 2*Ns, Ne]

        # Find the positions of the occupied sites
        A = jnp.take_along_axis(orbitals, R[:, :, None], axis=1) #* shape = [batch, Ne, Ne]

        log_det = _log_det(A)
        return log_det

class FermionicTransformer(nn.Module):
    num_layers: int
    d_model: int
    num_heads: int
    Ne: int
    Ns: int
    Nbands: int = 2
    dtype: Any = jnp.float64
    transl_invariant: bool = True
    two_dimensional: bool = True

    def setup(self):
        self.embed = nn.Embed(2**self.Nbands, self.d_model, embedding_init=nn.initializers.xavier_uniform(), param_dtype=self.dtype)

        self.encoder = Encoder(num_layers=self.num_layers,
                               d_model=self.d_model,
                               num_heads=self.num_heads,
                               seq_len=self.Ns,
                               dtype=self.dtype,
                               transl_invariant=self.transl_invariant,
                               two_dimensional=self.two_dimensional)

        self.out_layer_norm = nn.LayerNorm(param_dtype=self.dtype)

        self.output_layer = OutputHeadDet(d_model=self.d_model,
                                           Ne=self.Ne, 
                                           Ns=self.Ns, 
                                           dtype=self.dtype,
                                           Nbands=self.Nbands)

    def __call__(self, state):
        state = jax.tree.map(jnp.atleast_2d, state)
        n = state.config

        # Embedding
        n_up = n[..., :state.Ns, None]
        n_down = n[..., state.Ns:, None]
        n = jnp.concatenate((n_up, n_down), axis=-1)
        n = jnp.sum(n * 2**jnp.arange(self.Nbands), axis=-1)
        x = self.embed(n)

        # Encoder
        x = self.encoder(x)

        # output layer norm
        y = self.out_layer_norm(x)  # [Ns, d_model]

        f_non_zero = lambda x : x.nonzero(size=state.Ne)[0]
        R = jax.vmap(f_non_zero)(state.config) #* shape = [batch, Ne]
        log_amps = self.output_layer(y, R) #* notice the skip connection

        return log_amps