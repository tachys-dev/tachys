import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tachys.lattice.fermions.fermion_state import FermionState
from tachys.lattice.lattice_database import chain, square
from tachys.lattice.lattice_symmetries import point_group, translation_group
from tachys.lattice.spins.spin_state import SpinState
from tachys.lattice.state_array import get_array, replace_array
from tachys.lattice.symmetries import (
    combine_perm_groups,
    expand_perm,
    fermionic_sign,
    invert_perm,
    sign_permutation,
    symmetrize_wf,
)


def _cycle_sign(perm):
    """Independent parity reference via cycle decomposition (not inversion
    counting): sign = product over cycles of (-1)^(cycle_length - 1)."""
    n = len(perm)
    visited = [False] * n
    sign = 1
    for i in range(n):
        if visited[i]:
            continue
        cycle_len = 0
        j = i
        while not visited[j]:
            visited[j] = True
            j = perm[j]
            cycle_len += 1
        if cycle_len % 2 == 0:
            sign *= -1
    return sign


def _toy_apply(params, state):
    """Deliberately position-dependent (not already symmetric), so wrapping
    it is a non-trivial check: sum_i w_i * config_i."""
    return jnp.sum(params["w"] * get_array(state), axis=-1).astype(jnp.complex128)


def _zero_apply(params, state):
    return jnp.zeros(get_array(state).shape[0], dtype=jnp.complex128)


@pytest.fixture(scope="module")
def square_lat():
    return square(shape=(4, 4))


@pytest.fixture(scope="module")
def chain_lat():
    return chain(4)


# ─── invert_perm ─────────────────────────────────────────────────────────────

def test_invert_perm_is_true_functional_inverse_translations(square_lat):
    perms, _ = translation_group(square_lat)
    perms_inv = invert_perm(perms)
    Ns = perms.shape[-1]
    for perm, perm_inv in zip(perms, perms_inv):
        assert np.array_equal(perm[perm_inv], np.arange(Ns))
        assert np.array_equal(perm_inv[perm], np.arange(Ns))


def test_invert_perm_is_true_functional_inverse_point_group(square_lat):
    ops = point_group(square_lat)
    perms = np.stack([op.perm for op in ops])
    perms_inv = invert_perm(perms)
    Ns = perms.shape[-1]
    for perm, perm_inv in zip(perms, perms_inv):
        assert np.array_equal(perm[perm_inv], np.arange(Ns))


def test_invert_perm_handles_single_row(chain_lat):
    perms, _ = translation_group(chain_lat)
    perm = perms[1]
    perm_inv = invert_perm(perm)
    assert perm_inv.shape == perm.shape
    assert np.array_equal(perm[perm_inv], np.arange(perm.shape[-1]))


def test_invert_perm_closed_under_group_translations(square_lat):
    """Every inverse permutation is itself one of the group's own rows."""
    perms, _ = translation_group(square_lat)
    perms_inv = invert_perm(perms)
    perm_rows = {tuple(row) for row in perms.tolist()}
    for row in perms_inv.tolist():
        assert tuple(row) in perm_rows


def test_invert_perm_closed_under_group_point_group(square_lat):
    ops = point_group(square_lat)
    perms = np.stack([op.perm for op in ops])
    perms_inv = invert_perm(perms)
    perm_rows = {tuple(row) for row in perms.tolist()}
    for row in perms_inv.tolist():
        assert tuple(row) in perm_rows


def test_invert_perm_hand_example():
    # Ns=4, shift-by-one translation, matches translation_group(chain(4))[1].
    perm = np.array([1, 2, 3, 0])
    assert np.array_equal(invert_perm(perm), np.array([3, 0, 1, 2]))


# ─── expand_perm ─────────────────────────────────────────────────────────────

def test_expand_perm_is_noop_at_n_bands_one():
    perm = np.array([1, 2, 3, 0])
    assert np.array_equal(expand_perm(perm, 1), perm)


def test_expand_perm_shape_and_band_slices():
    perm = np.array([1, 2, 3, 0])
    Ns = perm.shape[-1]
    for n_bands in (2, 3):
        expanded = expand_perm(perm, n_bands)
        assert expanded.shape == (n_bands * Ns,)
        for b in range(n_bands):
            band_slice = expanded[b * Ns:(b + 1) * Ns]
            assert np.array_equal(band_slice, perm + b * Ns)


def test_expand_perm_batched_input():
    perms, _ = translation_group(chain(4))
    expanded = expand_perm(perms, n_bands=2)
    Ns = perms.shape[-1]
    assert expanded.shape == (perms.shape[0], 2 * Ns)
    for t in range(perms.shape[0]):
        for b in range(2):
            assert np.array_equal(expanded[t, b * Ns:(b + 1) * Ns], perms[t] + b * Ns)


# ─── fermionic_sign ──────────────────────────────────────────────────────────

def test_fermionic_sign_identity_is_plus_one():
    for n in (1, 2, 4, 6):
        perm = jnp.arange(n)
        assert fermionic_sign(perm) == 1


def test_fermionic_sign_single_transposition_is_minus_one():
    perm = jnp.array([1, 0, 2, 3])
    assert fermionic_sign(perm) == -1


def test_fermionic_sign_matches_cycle_decomposition_reference():
    rng = np.random.default_rng(0)
    for n in (3, 4, 5, 6):
        for _ in range(10):
            perm = rng.permutation(n)
            expected = _cycle_sign(perm.tolist())
            assert fermionic_sign(jnp.array(perm)) == expected


def test_fermionic_sign_matches_reference_over_all_permutations_of_4():
    for perm in itertools.permutations(range(4)):
        expected = _cycle_sign(list(perm))
        assert fermionic_sign(jnp.array(perm)) == expected


def test_fermionic_sign_ignores_sentinel_padding():
    # Real entries {0, 2}, padded with a sentinel (5) larger than every real
    # value in the trailing slots — padding must not register as inversions.
    perm_no_pad = jnp.array([2, 0])
    perm_padded = jnp.array([2, 0, 5, 5])
    assert fermionic_sign(perm_no_pad) == fermionic_sign(perm_padded)


# ─── sign_permutation ────────────────────────────────────────────────────────

def test_sign_permutation_hand_computed_two_particle_example():
    # Ns=4, config occupied at {0,2}, perm = shift-by-one = translation_group(chain(4))[1].
    perm = np.array([1, 2, 3, 0])
    perm_inv = invert_perm(perm)  # [3, 0, 1, 2]
    config = jnp.array([1, 0, 1, 0])
    # config[..., perm] moves the occupied set {0,2} -> {1,3}; relabeling the
    # two fermions by perm_inv gives the sequence (perm_inv[0], perm_inv[2])
    # = (3, 1), one inversion -> sign = -1.
    assert sign_permutation(config, perm_inv) == -1


def test_sign_permutation_matches_translation_group_row():
    perms, _ = translation_group(chain(4))
    perm = perms[1]
    assert np.array_equal(perm, np.array([1, 2, 3, 0]))
    perm_inv = invert_perm(perm)
    config = jnp.array([1, 0, 1, 0])
    assert sign_permutation(config, perm_inv) == -1


def test_sign_permutation_identity_perm_is_always_plus_one():
    perm_inv = jnp.arange(4)
    for occ in itertools.combinations(range(4), 2):
        config = jnp.zeros(4).at[jnp.array(occ)].set(1)
        assert sign_permutation(config, perm_inv) == 1


def test_sign_permutation_generalizes_over_bands():
    perm = np.array([1, 2, 3, 0])
    perm_inv = jnp.array(invert_perm(perm))

    band0 = jnp.array([1, 0, 1, 0])  # occ {0,2} -> sign -1 (hand example above)
    band1 = jnp.array([0, 1, 0, 1])  # occ {1,3} -> sign +1 (hand-derived in plan)
    band2 = jnp.array([1, 1, 1, 1])  # occ {0,1,2,3} -> sign -1 (hand-derived in plan)

    per_band_signs = [sign_permutation(band, perm_inv) for band in (band0, band1, band2)]
    assert per_band_signs == pytest.approx([-1, 1, -1])

    full_config = jnp.concatenate([band0, band1, band2])
    total_sign = sign_permutation(full_config, perm_inv)
    expected = 1.0
    for s in per_band_signs:
        expected *= s
    assert total_sign == pytest.approx(expected)


def test_sign_permutation_vmaps_over_batch():
    perm = np.array([1, 2, 3, 0])
    perm_inv = jnp.array(invert_perm(perm))
    configs = jnp.array([[1, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
    signs = jax.vmap(sign_permutation, in_axes=(0, None))(configs, perm_inv)
    assert signs.shape == (3,)
    assert signs[0] == -1
    assert signs[2] == 1  # empty occupation: trivially sign +1


# ─── symmetrize_wf (spins) ───────────────────────────────────────────────────

def test_symmetrize_wf_invariant_under_translation():
    lat = chain(4)
    perms, _ = translation_group(lat)
    state = SpinState(spins=jnp.array([[1., 2., 3., 4.]]), lattice=lat)
    params = {"w": jnp.array([0.1, 0.2, -0.15, 0.05])}

    wrapped = symmetrize_wf(_toy_apply, perms)
    f0 = wrapped(params, state)

    for t in range(perms.shape[0]):
        state_t = replace_array(state, get_array(state)[..., perms[t]])
        f_t = wrapped(params, state_t)
        assert jnp.allclose(f0, f_t)


def test_symmetrize_wf_invariant_under_point_group():
    lat = square(shape=(4, 4))
    ops = point_group(lat)
    perms = jnp.array(np.stack([op.perm for op in ops]))

    rng = np.random.default_rng(0)
    state = SpinState(spins=jnp.array(rng.normal(size=(1, 16))), lattice=lat)
    params = {"w": jnp.array(rng.normal(size=16))}

    wrapped = symmetrize_wf(_toy_apply, perms)
    f0 = wrapped(params, state)

    for t in range(perms.shape[0]):
        state_t = replace_array(state, get_array(state)[..., perms[t]])
        f_t = wrapped(params, state_t)
        assert jnp.allclose(f0, f_t), ops[t].name


def test_symmetrize_wf_matches_brute_force_combination():
    lat = chain(4)
    perms, _ = translation_group(lat)
    state = SpinState(spins=jnp.array([[1., 2., 3., 4.], [5., -1., 2., 0.]]), lattice=lat)
    params = {"w": jnp.array([0.1, 0.2, -0.15, 0.05])}

    wrapped = symmetrize_wf(_toy_apply, perms)
    actual = wrapped(params, state)

    raw = jnp.stack([
        _toy_apply(params, replace_array(state, get_array(state)[..., perms[t]]))
        for t in range(perms.shape[0])
    ], axis=0)
    zeroth = raw[0]
    expected = zeroth + jnp.log(jnp.sum(jnp.exp(raw - zeroth), axis=0))
    assert jnp.allclose(actual, expected)


def test_symmetrize_wf_sector_chars_matches_hand_formula():
    lat = chain(2)
    perms, _ = translation_group(lat)  # order 2: [identity, swap]
    state = SpinState(spins=jnp.array([[3., -7.]]), lattice=lat)
    params = {"w": jnp.array([2.0, 5.0])}
    sector_chars = jnp.array([0., 1.])

    wrapped = symmetrize_wf(_toy_apply, perms, sector_chars=sector_chars)
    actual = wrapped(params, state)

    f0 = _toy_apply(params, state)
    f1 = _toy_apply(params, replace_array(state, get_array(state)[..., perms[1]]))
    expected = f0 + jnp.log(1. + jnp.exp(1j * jnp.pi * sector_chars[1] + f1 - f0))
    assert jnp.allclose(actual, expected)


# ─── combine_perm_groups ──────────────────────────────────────────────────────

def test_combine_perm_groups_matches_nested_symmetrize_wf():
    lat = square(shape=(4, 4))
    coset_perms, _ = translation_group(lat)
    point_perms = np.stack([op.perm for op in point_group(lat)])

    rng = np.random.default_rng(0)
    state = SpinState(spins=jnp.array(rng.normal(size=(1, 16))), lattice=lat)
    params = {"w": jnp.array(rng.normal(size=16))}

    nested = symmetrize_wf(symmetrize_wf(_toy_apply, coset_perms), point_perms)
    expected = nested(params, state)

    combined_perms, combined_chars = combine_perm_groups(coset_perms, point_perms)
    assert combined_chars is None
    actual = symmetrize_wf(_toy_apply, combined_perms)(params, state)

    assert jnp.allclose(actual, expected)


def test_combine_perm_groups_matches_nested_symmetrize_wf_with_sector_chars():
    lat = square(shape=(4, 4))
    coset_perms, _ = translation_group(lat)
    point_perms = np.stack([op.perm for op in point_group(lat)])

    rng = np.random.default_rng(1)
    state = SpinState(spins=jnp.array(rng.normal(size=(1, 16))), lattice=lat)
    params = {"w": jnp.array(rng.normal(size=16))}
    chars_a = rng.integers(0, 2, size=coset_perms.shape[0]).astype(float)
    chars_b = rng.integers(0, 2, size=point_perms.shape[0]).astype(float)

    nested = symmetrize_wf(
        symmetrize_wf(_toy_apply, coset_perms, sector_chars=chars_a),
        point_perms, sector_chars=chars_b,
    )
    expected = nested(params, state)

    combined_perms, combined_chars = combine_perm_groups(
        coset_perms, point_perms, chars_a=chars_a, chars_b=chars_b)
    actual = symmetrize_wf(_toy_apply, combined_perms, sector_chars=combined_chars)(params, state)

    assert jnp.allclose(actual, expected)


def test_combine_perm_groups_rows_are_bijections():
    lat = square(shape=(4, 4))
    coset_perms, _ = translation_group(lat)
    point_perms = np.stack([op.perm for op in point_group(lat)])
    Ns = coset_perms.shape[-1]

    combined_perms, _ = combine_perm_groups(coset_perms, point_perms)
    assert combined_perms.shape == (coset_perms.shape[0] * point_perms.shape[0], Ns)
    for row in combined_perms:
        assert np.array_equal(np.sort(row), np.arange(Ns))


# ─── symmetrize_wf (fermions) ────────────────────────────────────────────────
#
# For fermions the projected amplitude is NOT literally constant across a
# group orbit: a genuine sector eigenstate transforms as
#   ψ_sym(g.σ) = [sign(g,σ) / chi(g)] * ψ_sym(σ)
# (the same fermionic sign every individual term picks up under g). With
# trivial characters this is
#   f_sym(g.σ) = f_sym(σ) + i*pi * (sign(g,σ) < 0),
# which is the invariant checked below (not naive equality). Note
# symmetrize_wf detects the FermionState and adds this sign automatically -
# the caller only ever supplies the single forward `perms` array.

def test_symmetrize_wf_fermions_signed_invariance_under_translation():
    lat = chain(4)
    base_perms, _ = translation_group(lat)
    perms = jnp.array(expand_perm(np.asarray(base_perms), n_bands=2))
    perms_inv = jnp.array(invert_perm(np.asarray(base_perms)))  # independent reference only

    band0 = jnp.array([1., 0., 1., 0.])
    band1 = jnp.array([0., 1., 0., 1.])
    config = jnp.concatenate([band0, band1])[None, :]
    state = FermionState(occupations=config, lattice=lat, Ne=2, Nbands=2)
    params = {"w": jnp.array([0.1, 0.2, 0.05, 0.3, 0.15, 0.25, 0.02, 0.4])}

    wrapped = symmetrize_wf(_toy_apply, perms)
    f0 = wrapped(params, state)

    for t in range(perms.shape[0]):
        state_t = replace_array(state, get_array(state)[..., perms[t]])
        f_t = wrapped(params, state_t)
        sign_t = sign_permutation(get_array(state)[0], perms_inv[t])
        predicted = f0 + 1j * jnp.pi * (sign_t < 0)
        assert jnp.allclose(f_t, predicted)


def test_symmetrize_wf_fermions_matches_brute_force_combination():
    lat = chain(4)
    base_perms, _ = translation_group(lat)
    perms = jnp.array(expand_perm(np.asarray(base_perms), n_bands=2))
    perms_inv = jnp.array(invert_perm(np.asarray(base_perms)))  # independent reference only

    band0 = jnp.array([1., 0., 1., 0.])
    band1 = jnp.array([0., 1., 0., 1.])
    config = jnp.concatenate([band0, band1])[None, :]
    state = FermionState(occupations=config, lattice=lat, Ne=2, Nbands=2)
    params = {"w": jnp.array([0.1, 0.2, 0.05, 0.3, 0.15, 0.25, 0.02, 0.4])}

    wrapped = symmetrize_wf(_toy_apply, perms)
    actual = wrapped(params, state)

    raw = []
    for t in range(perms.shape[0]):
        permuted = replace_array(state, get_array(state)[..., perms[t]])
        log_amp = _toy_apply(params, permuted)
        sign_t = sign_permutation(get_array(state)[0], perms_inv[t])
        raw.append(log_amp + 1j * jnp.pi * (sign_t < 0))
    raw = jnp.stack(raw, axis=0)
    zeroth = raw[0]
    expected = zeroth + jnp.log(jnp.sum(jnp.exp(raw - zeroth), axis=0))
    assert jnp.allclose(actual, expected)


def test_symmetrize_wf_fermions_sign_phase_isolated():
    # M=1: the combine step is trivial (zeroth + log(exp(0)) == zeroth), so
    # the output is exactly wf_apply(...) + i*pi*(sign<0), isolating the
    # sign-wiring end to end. Reuses the hand-derived Ns=4 example above:
    # occupied {0,2}, perm=[1,2,3,0] -> sign=-1.
    perm = np.array([1, 2, 3, 0])
    perms = jnp.array(perm)[None, :]

    lat = chain(4)
    config = jnp.array([[1., 0., 1., 0.]])
    state = FermionState(occupations=config, lattice=lat, Ne=2, Nbands=1)

    wrapped = symmetrize_wf(_zero_apply, perms)
    out = wrapped({}, state)
    assert jnp.allclose(out, 1j * jnp.pi)


def test_symmetrize_wf_auto_detects_spin_vs_fermion_on_identical_perms():
    """SpinState gets no sign phase; FermionState with the same occupied
    pattern and the same odd permutation gets exactly +i*pi added."""
    perm = np.array([1, 2, 3, 0])
    perms = jnp.array(perm)[None, :]
    lat = chain(4)

    spin_state = SpinState(spins=jnp.array([[1., 0., 1., 0.]]), lattice=lat)
    fermion_state = FermionState(occupations=jnp.array([[1., 0., 1., 0.]]), lattice=lat, Ne=2, Nbands=1)

    wrapped = symmetrize_wf(_zero_apply, perms)
    spin_out = wrapped({}, spin_state)
    fermion_out = wrapped({}, fermion_state)

    assert jnp.allclose(spin_out, 0.0)
    assert jnp.allclose(fermion_out, spin_out + 1j * jnp.pi)


def test_symmetrize_wf_fermions_matches_under_jit():
    # Regression: routing perms through numpy's argsort dispatch inside the
    # closure breaks under jax.jit tracing (numpy delegates to jax's own
    # argsort, which errors when `perms` is being traced) - the internal
    # inverse must use jnp.argsort, not the host-side invert_perm.
    perm = np.array([1, 2, 3, 0])
    perms = jnp.array(perm)[None, :]
    lat = chain(4)
    state = FermionState(occupations=jnp.array([[1., 0., 1., 0.]]), lattice=lat, Ne=2, Nbands=1)
    params = {"w": jnp.array([0.1, 0.2, 0.3, 0.4])}

    wrapped = symmetrize_wf(_toy_apply, perms)
    eager = wrapped(params, state)
    jitted = jax.jit(wrapped)(params, state)
    assert jnp.allclose(eager, jitted)
