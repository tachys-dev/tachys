"""The frozen configurations in `testing_configs` must stay physical.

They replace calls to `init_config_fixed_magn` / `init_config_spinful`, so
nothing else in the suite would notice if one were truncated or corrupted.
These checks re-derive the invariants those generators guarantee.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from testing_configs import CONFIG_SPECS, frozen_config


@pytest.mark.parametrize("name", sorted(CONFIG_SPECS))
def test_frozen_config_invariants(name):
    spec = CONFIG_SPECS[name]
    cfg = np.asarray(frozen_config(name))

    assert cfg.shape == spec["shape"]

    if spec["kind"] == "spins":
        assert set(np.unique(cfg)) <= {-1, 1}
        if spec["sz"] is not None:
            assert np.all(cfg.sum(axis=1) == spec["sz"])
    else:
        assert set(np.unique(cfg)) <= {0, 1}
        Ns = cfg.shape[1] // 2
        # Ne electrons in total, and a fixed Sz=0 split between the two spins.
        assert np.all(cfg.sum(axis=1) == spec["Ne"])
        assert np.all(cfg[:, :Ns].sum(axis=1) == spec["Ne"] // 2)
        assert np.all(cfg[:, Ns:].sum(axis=1) == spec["Ne"] // 2)

    # dtype override works, and the default is the recorded one.
    assert frozen_config(name, dtype=jnp.float64).dtype == jnp.float64
