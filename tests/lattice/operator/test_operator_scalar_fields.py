"""Scalar operator fields.

``_Operator.__post_init__`` turns every scalar pytree field into a ``(1,)``
array, so that ``__call__`` can vmap over the term axis. NumPy scalars -- such
as the ``np.int64`` entries of ``lattice.bonds`` output -- and Python complex
numbers must be converted like Python ints and floats, or that vmap fails on a
rank-0 leaf.
"""

import jax
import numpy as np
import pytest

from tachys.lattice.fermions.fermion_operators import Cup_dag, HoppingUp
from tachys.lattice.fermions.fermion_state import FermionState, init_config_spinful
from tachys.lattice.lattice_database import square
from tachys.lattice.spins.spin_operators import Sz, XYExchange
from tachys.lattice.spins.spin_state import SpinState, init_config_fixed_magn


@pytest.fixture(scope="module")
def spin_state():
    lat = square(shape=(4, 4))
    spins = init_config_fixed_magn(jax.random.key(0), lat.Ns, N_mc=4)
    return SpinState(spins=spins, lattice=lat)


@pytest.fixture(scope="module")
def fermion_state():
    lat = square(shape=(2, 2))
    occ, _, _ = init_config_spinful(jax.random.key(1), Ns=lat.Ns, Ne=2, N_mc=4)
    return FermionState(occupations=occ, Ne=2, lattice=lat)


@pytest.mark.parametrize("site", [np.int64(3), np.int32(3)])
def test_numpy_integer_site(spin_state, fermion_state, site):
    op = Sz(site)
    assert op.site.shape == (1,)
    np.testing.assert_array_equal(op(spin_state).matrix_element,
                                  Sz(3)(spin_state).matrix_element)

    r = Cup_dag(site)(fermion_state)
    r_ref = Cup_dag(3)(fermion_state)
    np.testing.assert_array_equal(r.matrix_element, r_ref.matrix_element)
    np.testing.assert_array_equal(r.mask, r_ref.mask)


def test_numpy_float32_coupling(spin_state):
    op = Sz(3, coupling=np.float32(0.5))
    assert op.coupling.shape == (1,)
    np.testing.assert_allclose(op(spin_state).matrix_element,
                               Sz(3, coupling=0.5)(spin_state).matrix_element)


def test_complex_coupling_matches_scalar_multiplication(fermion_state):
    """A complex hopping amplitude passed directly equals one multiplied in."""
    direct = HoppingUp(0, 1, coupling=1j)(fermion_state)
    scaled = (1j * HoppingUp(0, 1))(fermion_state)
    np.testing.assert_allclose(direct.matrix_element, scaled.matrix_element)
    np.testing.assert_array_equal(direct.mask, scaled.mask)


def test_single_bond_from_lattice_bonds(spin_state):
    """S_i . S_j on one bond taken from lattice.bonds.

    Merging two terms concatenates their fields into arrays, which hid the
    problem inside loops over many bonds; a term that is never merged keeps its
    np.int64 indices unless __post_init__ converts them.
    """
    src, dst = spin_state.lattice.bonds((1, 0))
    i, j = src[0], dst[0]                                   # np.int64
    r = (Sz(i) * Sz(j) + 0.5 * XYExchange(i, j))(spin_state)

    i, j = int(i), int(j)
    r_ref = (Sz(i) * Sz(j) + 0.5 * XYExchange(i, j))(spin_state)

    np.testing.assert_array_equal(r.diagonal.matrix_element,
                                  r_ref.diagonal.matrix_element)
    np.testing.assert_array_equal(r.offdiagonal.matrix_element,
                                  r_ref.offdiagonal.matrix_element)
    np.testing.assert_array_equal(r.offdiagonal.mask, r_ref.offdiagonal.mask)
    np.testing.assert_array_equal(r.offdiagonal.connected_states.spins,
                                  r_ref.offdiagonal.connected_states.spins)
