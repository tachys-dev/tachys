# =========================================================================== #
#  SYMMETRIES  (site-index permutations for momentum / point-group projection)
# =========================================================================== #
#
# USAGE OVERVIEW
# --------------
#   T, lab = translation_group(lat)      # (Nt, Ns) perms + (Nt, 2) cell shifts
#   ph     = momentum_phases(lat, m)     # characters aligned with T's rows
#   ops    = point_group(lat)            # named SymOps (rotations + reflections)
#
# CHOOSING A MOMENTUM SECTOR
#   Only translation_group + momentum_phases are needed. Form the projector
#       P_k = (1/Nt) * sum_t  ph[t] * T_t
#   from the permutations T and the phases ph for momentum m = (m1, m2), where
#   k = 2*pi*(m1/Lx, m2/Ly). The point group is the refinement: at a given k the
#   little group (point-group ops fixing k modulo a reciprocal-lattice vector)
#   supplies the extra label, built the same way from point_group() perms and the
#   appropriate irrep characters. These two steps are kept separate so the little
#   group can be assembled as needed. (A dedicated little_group(lat, m) helper
#   could filter point_group() to the ops preserving a given momentum — not yet
#   implemented.)
#
# PERMUTATION CONVENTION
#   perm[s] = g(s), the image of site s under the operation. Building a projector
#   sums group elements, and every group here is closed under inverse, so the
#   projector is convention-independent. But applying a SINGLE operation to a
#   config: config[perm] yields (g^{-1} . sigma), NOT (g . sigma). If you need
#   g . sigma, apply the inverse permutation (the perm of g^{-1}, which is also
#   present in the returned list).
#
# CORRECTNESS CAVEATS
#   * point_group fixes `center` (default the origin, where basis atom 0 sits).
#     For a group whose natural rotation center is NOT a basis-0 atom (e.g. a
#     Kagome plaquette center), pass the correct `center` or only the site-
#     symmetry subgroup of the origin will be detected.
#   * Detection is exact modulo the retrieve_index tolerance. An incompatible
#     cluster shape (e.g. a 6x4 triangular cluster that breaks 6-fold symmetry)
#     correctly returns the smaller compatible group rather than silently
#     including a broken operation.
#
# PERFORMANCE NOTE
#   The heavy loops (site permutation under each candidate point-group matrix,
#   and the translation permutations) run inside Numba kernels that call the
#   njit _retrieve_index directly, so there is no per-site Python<->Numba
#   boundary crossing. The naming helpers run only ~24 times and stay in Python.
# =========================================================================== #

from math import gcd
from typing import NamedTuple
import numpy as np
from numba import njit

from tachys.lattice.lattice import _retrieve_index
# NOTE: _retrieve_index is defined in lattice.py and must be importable/in scope.


class SymOp(NamedTuple):
    name: str
    matrix: np.ndarray   # 2x2 point-group matrix (identity for pure translations)
    perm:   np.ndarray   # [Ns] int: perm[s] = image of site s under the operation


# --------------------------------------------------------------------------- #
#  numba kernels                                                              #
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _translation_perms(site_coords, c2s, Lx, Ly, Nt1, Nt2):
    """
    Permutations for every cell shift (n1, n2), row order: n2 outer, n1 inner.
    Nt1/Nt2 = Lx/Ly for periodic directions, 1 otherwise.
    """
    Ns = site_coords.shape[0]
    Nt = Nt1 * Nt2
    perms = np.empty((Nt, Ns), dtype=np.int64)
    labels = np.empty((Nt, 2), dtype=np.int64)
    t = 0
    for n2 in range(Nt2):
        for n1 in range(Nt1):
            for s in range(Ns):
                ni = (site_coords[s, 0] + n2) % Ly
                nj = (site_coords[s, 1] + n1) % Lx
                perms[t, s] = c2s[ni, nj, site_coords[s, 2]]
            labels[t, 0] = n1
            labels[t, 1] = n2
            t += 1
    return perms, labels


@njit(cache=True)
def _perms_of_matrices(Os, points, center, Ainv, basis_frac, c2s,
                       Lx, Ly, pbc_x, pbc_y, tol=1e-6):
    """
    For each 2x2 orthogonal matrix Os[m], the induced site permutation about
    `center`. Returns perms[M, Ns] (-1 where the map leaves the lattice) and
    valid[M] (True iff the map is a bijection of the sites).
    """
    M = Os.shape[0]
    Ns = points.shape[0]
    perms = np.full((M, Ns), -1, dtype=np.int64)
    valid = np.zeros(M, dtype=np.bool_)
    seen = np.zeros(Ns, dtype=np.bool_)

    for m in range(M):
        ok = True
        for s in range(Ns):
            dx = points[s, 0] - center[0]
            dy = points[s, 1] - center[1]
            # p = O @ (point - center) + center
            px_ = Os[m, 0, 0] * dx + Os[m, 0, 1] * dy + center[0]
            py_ = Os[m, 1, 0] * dx + Os[m, 1, 1] * dy + center[1]
            # frac = p @ Ainv   (rows of A are a1, a2)
            f0 = px_ * Ainv[0, 0] + py_ * Ainv[1, 0]
            f1 = px_ * Ainv[0, 1] + py_ * Ainv[1, 1]
            idx = _retrieve_index(f0, f1, basis_frac, c2s, Lx, Ly, pbc_x, pbc_y, tol)
            if idx < 0:
                ok = False
                break
            perms[m, s] = idx
        if not ok:
            continue
        # bijection check (reset the scratch buffer each time)
        for s in range(Ns):
            seen[s] = False
        bij = True
        for s in range(Ns):
            v = perms[m, s]
            if seen[v]:
                bij = False
                break
            seen[v] = True
        valid[m] = bij
    return perms, valid


# --------------------------------------------------------------------------- #
#  translation group  (this is what selects momentum)                         #
# --------------------------------------------------------------------------- #
def translation_group(lat):
    """
    Full translation group of the finite cluster under PBC.

    Returns
    -------
    perms  : (Nt, Ns) int64. perms[t, s] = image of site s under translation t,
             i.e. the site whose cell is (i+n2, j+n1) mod (Ly, Lx), same sublattice.
    labels : (Nt, 2) int64. (n1, n2) = cell shift along (a1, a2) for each t.

    A non-periodic direction contributes only n = 0. The group is abelian; its
    irreps are momenta.
    """
    Lx, Ly = lat.L
    px, py = lat.pbc
    Nt1 = Lx if px else 1
    Nt2 = Ly if py else 1
    return _translation_perms(lat.site_coords, lat.cell_to_site, Lx, Ly, Nt1, Nt2)


def momentum_phases(lat, m):
    """
    Characters of the translation group for momentum sector m = (m1, m2),
    aligned with the rows of `translation_group(lat)[0]`.

    k = 2*pi*(m1/Lx, m2/Ly);  phase_t = exp(-i k . R_t).

    Projector onto momentum k:  P_k = (1/Nt) * sum_t phases[t] * T_t, which selects
    states with T_t |psi> = exp(+i k.R_t) |psi>. Flip the sign in the exponent
    (or negate m) for the opposite convention.
    """
    Lx, Ly = lat.L
    px, py = lat.pbc
    Nt1 = Lx if px else 1
    Nt2 = Ly if py else 1
    m1, m2 = m
    phases = np.empty(Nt1 * Nt2, dtype=np.complex128)
    t = 0
    for n2 in range(Nt2):                # same row order as translation_group
        for n1 in range(Nt1):
            phases[t] = np.exp(-2j * np.pi * (m1 * n1 / Lx + m2 * n2 / Ly))
            t += 1
    return phases


# --------------------------------------------------------------------------- #
#  point group  (auto-detected, Schoenflies-named)                            #
# --------------------------------------------------------------------------- #
def _name_rotation(O, n_rot):
    theta = np.arctan2(O[1, 0], O[0, 0]) % (2 * np.pi)
    p = int(round(theta * n_rot / (2 * np.pi))) % n_rot
    if p == 0:
        return "E"
    g = gcd(p, n_rot)
    n, pw = n_rot // g, p // g
    return f"C{n}" if pw == 1 else f"C{n}^{pw}"


def _name_reflection(O, lat, tol=1e-6):
    phi = 0.5 * np.arctan2(O[0, 1], O[0, 0])           # mirror-axis angle
    axis = np.array([np.cos(phi), np.sin(phi)])
    deg = int(round(np.degrees(phi))) % 180
    a1 = lat.lattice_vectors[0] / np.linalg.norm(lat.lattice_vectors[0])
    a2 = lat.lattice_vectors[1] / np.linalg.norm(lat.lattice_vectors[1])
    aligned = any(abs(abs(axis @ v) - 1) < 1e-3 for v in (a1, a2))
    return f"σv({deg}°)" if aligned else f"σd({deg}°)"


def point_group(lat, center=(0.0, 0.0), n_candidates=12):
    """
    Point-group symmetries of the finite cluster that fix `center`, returned as
    named SymOps. Rotations are named C_n^p (reduced to lowest terms, so a 60°
    rotation in a hexagonal group is C6, a 120° one is C3, 180° is C2, ...);
    reflections are σv (axis along a lattice vector) or σd (diagonal), tagged with
    the axis angle. Whatever subgroup is compatible with the cluster is detected
    automatically: C4v for square, C6v for triangular, lower for incompatible sizes.

    `center` is the fixed point of the rotations (default the origin, where basis
    atom 0 sits; see module caveat for centers that are not a basis-0 atom).
    `n_candidates` sets the angular resolution of the search grid (12 -> 30° steps,
    enough for 2-/3-/4-/6-fold axes).
    """
    thetas = [2 * np.pi * k / n_candidates for k in range(n_candidates)]
    rots = [np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]) for t in thetas]
    # reflections: rotate the x-axis mirror through every candidate angle
    Mx = np.array([[1.0, 0.0], [0.0, -1.0]])
    refls = [R @ Mx for R in rots]

    all_mats = rots + refls
    Os = np.ascontiguousarray(np.stack(all_mats))                  # (M, 2, 2)
    dets = np.array([np.sign(round(np.linalg.det(O))) for O in all_mats])

    Ainv = np.ascontiguousarray(np.linalg.inv(lat.lattice_vectors))
    px, py = lat.pbc
    perms, valid = _perms_of_matrices(
        Os, lat.points, np.asarray(center, dtype=float), Ainv,
        lat.basis_frac, lat.cell_to_site, lat.L[0], lat.L[1], px, py)

    n_rot = int(np.sum(valid & (dets > 0)))            # order of principal axis

    ops, seen = [], set()
    for m in range(len(all_mats)):
        if not valid[m]:
            continue
        O = Os[m]
        name = _name_rotation(O, n_rot) if dets[m] > 0 else _name_reflection(O, lat)
        key = perms[m].tobytes()
        if key in seen:                                # dedup coincident matrices
            continue
        seen.add(key)
        ops.append(SymOp(name=name, matrix=O, perm=perms[m].copy()))
    return ops