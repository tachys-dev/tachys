"""Operator-algebra merge rules.

``_Operator.__add__`` fuses two operators into a single batched term through a
flax state dict, which carries only ``pytree_node=True`` fields. Merging two
instances that differ in a *static* field would therefore keep only the left
operand's value and silently build a wrong operator -- these tests pin the
guard that keeps such terms apart.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from tachys.lattice.exact_diag import exact_diag, fermions_hilbert_space
from tachys.lattice.fermions.fermion_operators import (
    Hopping, HoppingUp, HoppingDown, N, Nup, Ndn,
)
from tachys.lattice.fermions.fermion_state import FermionState
from tachys.lattice.lattice_database import square
from tachys.lattice.operator.base import _OperatorSum
from tachys.lattice.spins.spin_operators import Sz


# ---------------------------------------------------------------- algebra ----

def test_mismatched_static_field_does_not_merge():
    """Hopping(band=0) + Hopping(band=2) must keep both terms, bands intact."""
    s = Hopping(i=0, j=1, band=0) + Hopping(i=2, j=3, band=2)

    assert isinstance(s, _OperatorSum)
    assert len(s.operators) == 2
    assert [op.band for op in s.operators] == [0, 2]
    assert s.operators[0].i == 0 and s.operators[0].j == 1
    assert s.operators[1].i == 2 and s.operators[1].j == 3


def test_matching_static_field_still_merges():
    """The guard must not over-refuse: same band still fuses into one term."""
    m = Hopping(i=0, j=1, band=2) + Hopping(i=5, j=6, band=2)

    assert type(m) is Hopping
    assert m.band == 2
    np.testing.assert_array_equal(m.i, [0, 5])
    np.testing.assert_array_equal(m.j, [1, 6])


def test_batched_and_scalar_coupling_merge():
    """same_treedef ignores leaf shapes, so batched + scalar must still fuse."""
    a = HoppingUp(np.arange(8), np.arange(1, 9), coupling=-np.ones(8))
    b = HoppingUp(3, 4, coupling=-0.5)
    m = a + b

    assert type(m) is HoppingUp
    assert m.i.shape == (9,)
    assert m.coupling.shape == (9,)
    np.testing.assert_allclose(m.coupling[-1], -0.5)


def test_operator_sum_buckets_by_static_field():
    """Adding into an existing sum merges onto the term with the same band."""
    s = Hopping(i=0, j=1, band=0) + Hopping(i=2, j=3, band=2)
    s = s + Hopping(i=4, j=5, band=0)

    assert isinstance(s, _OperatorSum)
    assert len(s.operators) == 2
    assert [op.band for op in s.operators] == [0, 2]
    np.testing.assert_array_equal(s.operators[0].i, [0, 4])   # batched onto band 0
    np.testing.assert_array_equal(s.operators[1].i, [2])      # band 2 untouched


def test_distinct_subclasses_still_do_not_merge():
    """The band-fixed subclass path is unchanged: different types never fuse."""
    s = HoppingUp(0, 1) + HoppingDown(0, 1)
    assert isinstance(s, _OperatorSum)
    assert len(s.operators) == 2

    s = Nup(0) + Ndn(0)
    assert isinstance(s, _OperatorSum)
    assert len(s.operators) == 2


def test_operators_without_static_fields_unaffected():
    """Spin operators carry no static fields and must merge as before."""
    m = Sz(0) + Sz(1)
    assert type(m) is Sz
    np.testing.assert_array_equal(m.site, [0, 1])

    m = N(0, band=1) + N(3, band=1)
    assert type(m) is N
    np.testing.assert_array_equal(m.site, [0, 3])


# ---------------------------------------------------------------- physics ----

L = 2
Ne = 2
U = 4.0


def _pack(state):
    occ = np.asarray(state.occupations)
    return (occ * 2 ** np.arange(occ.shape[-1])).sum(axis=-1)


def _hubbard(lat, hopping_cls_up, hopping_cls_dn, band_kwargs):
    """Hubbard H built from whichever pair of hopping classes is handed in."""
    src, dst = [], []
    for d in ((1, 0), (0, 1)):
        s, t = lat.bonds(d)
        src.append(s)
        dst.append(t)
    src = np.concatenate(src)
    dst = np.concatenate(dst)
    hop = np.ones(len(src))

    H = (hopping_cls_up(src, dst, coupling=-hop, **band_kwargs[0])
         + hopping_cls_dn(src, dst, coupling=-hop, **band_kwargs[1]))

    sites = np.arange(lat.Ns)
    H = H + Nup(sites, coupling=np.full(lat.Ns, U)) * Ndn(sites, coupling=np.ones(lat.Ns))
    return H


@pytest.fixture(scope="module")
def spectra():
    """Spectrum of the same Hubbard model built two ways.

    Reference: the band-fixed subclasses (different types, never merged).
    Generic:   bare Hopping with an explicit band= (same type, merge refused
               only because of the static-field guard).

    Without the guard the generic build fuses both terms onto band 0, giving
    duplicated spin-up bonds and no spin-down hopping -- a different spectrum.

    fermions_hilbert_space spans every Sz sector; that is fine here because
    this compares energies only (do not reuse for expectation values).
    """
    lat = square(shape=(L, L))
    basis = fermions_hilbert_space(lat.Ns, Ne)
    state_full = FermionState(
        occupations=jnp.array(basis, dtype=jnp.int8), lattice=lat, Ne=Ne
    )

    H_ref = _hubbard(lat, HoppingUp, HoppingDown, ({}, {}))
    H_gen = _hubbard(lat, Hopping, Hopping, ({"band": 0}, {"band": 1}))

    k = 6
    e_ref, _ = exact_diag(state_full, H_ref, _pack, k=k)
    e_gen, _ = exact_diag(state_full, H_gen, _pack, k=k)
    return np.asarray(e_ref), np.asarray(e_gen)


def test_generic_band_argument_builds_the_same_hamiltonian(spectra):
    e_ref, e_gen = spectra
    np.testing.assert_allclose(e_gen, e_ref, atol=1e-9)
