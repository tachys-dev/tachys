"""Initial Monte Carlo configurations used by the test suite.

Frozen for the same reason as the parameters in `testing_ansatz`: the suite's
hard-coded expectations (local energies, NTK entries, SR/SPRING updates,
sampled log-amplitudes) are all evaluated on these configurations, and JAX's
PRNG makes no promise that `init_config_fixed_magn(jax.random.key(1), ...)`
yields the same configurations across versions.  The recorded values are what
the generators produced when frozen (jax 0.7.1), so no expectation changes.

Unlike the parameters, these are *not* source literals: three of them hold
80k entries between them, so the whole set lives in `data/frozen_configs.npz`,
the way this suite already stores its larger reference arrays.  Every entry is
a plain +/-1 spin array or a 0/1 occupation array, and
`test_testing_configs.py` re-derives the invariants (fixed magnetization,
particle number) rather than trusting the file.

What is *not* frozen here: the Monte Carlo keys (`jax.random.split(key, N_mc)`)
handed to `sample`, `blur_states` and the actions.  Those are consumed inside
the library's own `jax.random` calls, so their configurations are only as
stable as the PRNG itself -- freezing the key would not change that.
"""
from pathlib import Path

import jax.numpy as jnp
import numpy as np

_NPZ_PATH = Path(__file__).parent / "data" / "frozen_configs.npz"
_cache = {}

# Each configuration and the invariant it must satisfy: `sz` is the fixed total
# magnetization of a spin configuration (None when it is unconstrained), `Ne`
# the electron number of an occupation configuration.
CONFIG_SPECS = {
    # spins, +/-1
    "square16_nmc16":                 dict(kind="spins", shape=(16, 16), sz=0),
    "local_estimator_square16":       dict(kind="spins", shape=(16, 16), sz=0),
    "blurred_square4_nmc8":           dict(kind="spins", shape=(8, 4), sz=0),
    "blurred_square4_nmc4096":        dict(kind="spins", shape=(4096, 4), sz=0),
    "blurred_square4_nmc8192_seed0":  dict(kind="spins", shape=(8192, 4), sz=0),
    "blurred_square4_nmc8192_seed1":  dict(kind="spins", shape=(8192, 4), sz=0),
    "blurred_chain6_nmc32":           dict(kind="spins", shape=(32, 6), sz=None),
    "lanczos_chain8_nmc32":           dict(kind="spins", shape=(32, 8), sz=None),
    "foundation_square16_nmc32":      dict(kind="spins", shape=(32, 16), sz=0),
    "foundation_operators_square4_a": dict(kind="spins", shape=(3, 4), sz=None),
    "foundation_operators_square4_b": dict(kind="spins", shape=(3, 4), sz=None),
    # spinful occupations, 0/1, ordered (up sites, down sites)
    "square16_ne4_nmc16":             dict(kind="occupations", shape=(16, 32), Ne=4),
    "square16_ne8_nmc32":             dict(kind="occupations", shape=(32, 32), Ne=8),
    "foundation_square16_ne10_nmc32": dict(kind="occupations", shape=(32, 32), Ne=10),
}


def frozen_config(name, dtype=None):
    """Return the frozen configuration registered under ``name``.

    ``dtype`` defaults to the dtype the generator produced (``int8`` for spins,
    ``int32`` for occupations, ``float64`` for the two drawn with
    ``jax.random.choice``/``bernoulli``); pass one to override it.
    """
    if name not in CONFIG_SPECS:
        raise KeyError(
            f"no frozen configuration named {name!r}; "
            f"available: {sorted(CONFIG_SPECS)}")
    if not _cache:
        with np.load(_NPZ_PATH) as f:
            _cache.update({k: f[k] for k in f.files})
    return jnp.asarray(_cache[name], dtype=dtype)
