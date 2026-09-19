"""The frozen parameters in `testing_ansatz` must stay consistent with its models.

`frozen_params` hard-codes the values so that no test depends on JAX's PRNG,
which means nothing would otherwise notice if an ansatz grew, lost or reshaped
a parameter.  This checks each tree against what that model's `init` actually
asks for -- structure and shapes only, never values.
"""
import jax
import jax.numpy as jnp
import pytest

from testing_ansatz import (
    FermionFoundationRBM, PARAM_SPECS, SpinRBM, frozen_params,
)
from tachys.lattice.fermions.hamiltonians.hubbard import hubbard_square_pbc
from tachys.lattice.foundation.foundation_state import FermionFoundationState
from tachys.lattice.foundation.operators import combine_systems, extract_system_couplings
from tachys.lattice.lattice_database import chain, square
from tachys.lattice.spins.spin_state import SpinState
from testing_configs import frozen_config


def _spin_rbm(spec):
    n_sites = spec["n_sites"]
    model = SpinRBM(hidden_units=spec["hidden_units"], dtype=jnp.float64,
                    complex=spec["complex"])
    dummy = SpinState(spins=jnp.ones((1, n_sites), dtype=jnp.float64),
                      lattice=chain(n_sites, pbc=True))
    return model, dummy


def _fermion_foundation_rbm(spec):
    L = int(round(spec["n_sites"] ** 0.5))
    n_systems = spec["n_systems"]
    occupations = frozen_config("foundation_square16_ne10_nmc32")
    N_mc = occupations.shape[0]
    H = combine_systems([hubbard_square_pbc(L, U=U) for U in range(n_systems)],
                        N_mc // n_systems)
    dummy = FermionFoundationState(
        occupations=occupations,
        lattice=square(shape=(L, L)),
        Ne=spec["Ne"],
        system_couplings=extract_system_couplings(H),
        system_ids=jnp.repeat(jnp.arange(n_systems), N_mc // n_systems),
        n_systems=n_systems,
    )
    return FermionFoundationRBM(hidden_units=spec["hidden_units"]), dummy


_BUILDERS = {"SpinRBM": _spin_rbm, "FermionFoundationRBM": _fermion_foundation_rbm}


@pytest.mark.parametrize("name", sorted(PARAM_SPECS))
def test_frozen_params_match_model(name):
    spec = PARAM_SPECS[name]
    model, dummy = _BUILDERS[spec["model"]](spec)

    ref = model.init(jax.random.key(0), dummy)
    got = frozen_params(name)

    assert jax.tree.structure(got) == jax.tree.structure(ref)
    assert jax.tree.map(jnp.shape, got) == jax.tree.map(jnp.shape, ref)
    assert all(x.dtype == jnp.float64 for x in jax.tree.leaves(got))

    # And the tree is usable as-is: a forward pass must run and be finite.
    assert jnp.all(jnp.isfinite(model.apply(got, dummy)))
