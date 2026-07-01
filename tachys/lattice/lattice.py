"""
lattice.py — geometry backend for VMC/NQS Hamiltonian construction.

(coordinate model, bond convention, and JAX note: see docstrings below)
"""

from typing import NamedTuple
import numpy as np
import jax.numpy as jnp
from numba import njit


# --------------------------------------------------------------------------- #
#  numba geometry primitives (operate on plain numpy arrays)                  #
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _build_sites(a1, a2, basis, Lx, Ly):
    """Return (points[Ns,2], coords[Ns,3]=(i,j,b), cell_to_site[Ly,Lx,nb])."""
    nb = basis.shape[0]
    Ns = Lx * Ly * nb
    points = np.zeros((Ns, 2))
    coords = np.zeros((Ns, 3), dtype=np.int64)          # (i along a2, j along a1, b)
    c2s = -np.ones((Ly, Lx, nb), dtype=np.int64)        # -1 marks an absent site
    idx = 0
    for i in range(Ly):                                 # row 0 sits at the a2 origin
        for j in range(Lx):
            ox = j * a1[0] + i * a2[0]
            oy = j * a1[1] + i * a2[1]
            for b in range(nb):
                points[idx, 0] = ox + basis[b, 0]
                points[idx, 1] = oy + basis[b, 1]
                coords[idx, 0] = i
                coords[idx, 1] = j
                coords[idx, 2] = b
                c2s[i, j, b] = idx
                idx += 1
    return points, coords, c2s


@njit(cache=True)
def _dist_matrix(points, sx_vec, sy_vec, pbc_x, pbc_y):
    """Minimum-image Euclidean distance matrix (for observables / plotting only)."""
    N = points.shape[0]
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            dx0 = points[i, 0] - points[j, 0]
            dy0 = points[i, 1] - points[j, 1]
            best = 1e18
            for sx in range(-1, 2):
                if sx != 0 and not pbc_x:
                    continue
                for sy in range(-1, 2):
                    if sy != 0 and not pbc_y:
                        continue
                    dx = dx0 + sx * sx_vec[0] + sy * sy_vec[0]
                    dy = dy0 + sx * sx_vec[1] + sy * sy_vec[1]
                    d = dx * dx + dy * dy
                    if d < best:
                        best = d
            D[i, j] = np.sqrt(best)
    return D


@njit(cache=True)
def _retrieve_index(c1, c2, basis_frac, c2s, Lx, Ly, pbc_x, pbc_y, tol=1e-6):
    """
    Low-level lookup. (c1, c2) = target coefficients in the (a1, a2) basis,
    INCLUDING any basis offset. Returns the site index, or -1 if no site exists
    there (OBC out of range, or the point does not coincide with any atom).
    """
    nb = basis_frac.shape[0]
    for b in range(nb):
        f1 = c1 - basis_frac[b, 0]
        f2 = c2 - basis_frac[b, 1]
        n1 = np.floor(f1 + 0.5)
        n2 = np.floor(f2 + 0.5)
        if abs(f1 - n1) < tol and abs(f2 - n2) < tol:
            j = int(n1)   # cells along a1
            i = int(n2)   # cells along a2
            if pbc_x:
                j %= Lx
            elif j < 0 or j >= Lx:
                continue
            if pbc_y:
                i %= Ly
            elif i < 0 or i >= Ly:
                continue
            return c2s[i, j, b]
    return -1


@njit(cache=True)
def _retrieve_batch(cs, basis_frac, c2s, Lx, Ly, pbc_x, pbc_y, tol=1e-6):
    out = np.empty(cs.shape[0], dtype=np.int64)
    for m in range(cs.shape[0]):
        out[m] = _retrieve_index(cs[m, 0], cs[m, 1], basis_frac, c2s,
                                 Lx, Ly, pbc_x, pbc_y, tol)
    return out


# --------------------------------------------------------------------------- #
#  Lattice (immutable NamedTuple)                                             #
# --------------------------------------------------------------------------- #
class Lattice(NamedTuple):
    """
    Immutable lattice geometry.

    COORDINATE MODEL
        A site is an integer triple (i, j, b): i = cell along a2 (row),
        j = cell along a1 (column), b = sublattice. `site_coords[s] = (i, j, b)`
        and the reverse map `cell_to_site[i, j, b] -> s` (-1 where absent) make
        every lookup pure integer / modular arithmetic.

    BOND CONVENTION  (see .bonds)
        .bonds(delta, b_from, b_to) connects sublattice b_from in cell C to
        sublattice b_to in cell C + delta, for every cell C. `delta` is a WHOLE-
        CELL displacement in (a1, a2) units; the intra-cell offset comes only
        from b_from / b_to.

    JAX NOTE
        A NamedTuple IS an automatic JAX pytree, so if you pass a Lattice into a
        jit-ed function JAX will try to flatten these numpy arrays into leaves.
        Don't — build the jnp bond-index arrays once with .bond_arrays() and pass
        ONLY those into traced code. Treat Lattice as host-side metadata.
    """
    lattice_vectors: np.ndarray   # rows are a1, a2
    basis:           np.ndarray   # Cartesian
    basis_frac:      np.ndarray   # in (a1, a2) units
    points:          np.ndarray   # [Ns, 2] Cartesian
    site_coords:     np.ndarray   # [Ns, 3] (i, j, b)
    cell_to_site:    np.ndarray   # [Ly, Lx, nb]
    dist_matrix:     np.ndarray   # [Ns, Ns]
    L:               tuple        # (Lx, Ly)
    Ns:              int
    nb:              int
    pbc:             tuple        # (pbc_x, pbc_y)

    # Identity-based, NOT value-based: lets a Lattice sit as static (pytree_node=False)
    # metadata on a State without JAX trying to hash/compare its numpy array fields.
    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    # ------------------------------------------------------------------ build #
    @classmethod
    def create(cls, a1, a2, basis, shape, pbc_x=True, pbc_y=True):
        """
        Build a lattice from unit-cell vectors, a Cartesian basis, and a shape.

        Parameters
        ----------
        a1, a2 : 2-vectors        primitive cell vectors
        basis  : (nb, 2) array    Cartesian positions of the atoms in one cell
        shape  : (Lx, Ly)         number of cells along a1 and a2
        pbc_x, pbc_y : bool       periodic boundaries along a1 / a2
        """
        a1 = np.ascontiguousarray(np.asarray(a1, dtype=float))
        a2 = np.ascontiguousarray(np.asarray(a2, dtype=float))
        basis = np.ascontiguousarray(np.atleast_2d(np.asarray(basis, dtype=float)))
        Lx, Ly = shape

        A = np.stack([a1, a2])                      # rows a1, a2
        basis_frac = basis @ np.linalg.inv(A)       # basis positions in (a1, a2) units

        points, coords, c2s = _build_sites(a1, a2, basis, Lx, Ly)
        D = _dist_matrix(points, Lx * a1, Ly * a2, pbc_x, pbc_y)

        return cls(
            lattice_vectors=A,
            basis=basis,
            basis_frac=basis_frac,
            points=points,
            site_coords=coords,
            cell_to_site=c2s,
            dist_matrix=np.round(D, 8),
            L=(Lx, Ly),
            Ns=points.shape[0],
            nb=basis.shape[0],
            pbc=(bool(pbc_x), bool(pbc_y)),
        )

    # -------------------------------------------------------------- lookups #
    def retrieve_index(self, c1, c2):
        """
        Low-level lookup: site index for target coefficients (c1, c2) in the
        (a1, a2) basis (basis offset included). Returns -1 if absent. Most code
        should use `bonds` / `neighbour_of` instead of calling this directly.
        """
        px, py = self.pbc
        return int(_retrieve_index(float(c1), float(c2), self.basis_frac,
                                   self.cell_to_site, self.L[0], self.L[1], px, py))

    def bonds(self, delta, b_from=0, b_to=None):
        """
        Directed (src, dst) index pairs for a cell displacement `delta`.

        A bond connects sublattice `b_from` in cell C to sublattice `b_to` in
        cell C + delta, for every cell C for which the target exists.

        Parameters
        ----------
        delta : (d1, d2)   CELL displacement in (a1, a2) units (whole cells; the
                           intra-cell offset comes from b_from/b_to, not delta)
        b_from : int       source sublattice (default 0)
        b_to   : int       target sublattice (default: same as b_from)

        Returns
        -------
        src, dst : int64 arrays of equal length. Under OBC, bonds whose target
                   falls outside the lattice are dropped.
        """
        if b_to is None:
            b_to = b_from
        d1, d2 = delta
        src = np.nonzero(self.site_coords[:, 2] == b_from)[0]
        cs = np.empty((src.size, 2))
        # target = source cell + delta + target-sublattice offset
        cs[:, 0] = self.site_coords[src, 1] + d1 + self.basis_frac[b_to, 0]  # along a1
        cs[:, 1] = self.site_coords[src, 0] + d2 + self.basis_frac[b_to, 1]  # along a2
        dst = _retrieve_batch(cs, self.basis_frac, self.cell_to_site,
                              self.L[0], self.L[1], *self.pbc)
        keep = dst >= 0
        return src[keep], dst[keep]

    def neighbour_of(self, site, delta, b_to=None):
        """
        Single site reached from `site`'s cell by cell displacement `delta`,
        landing on sublattice `b_to` (default: same sublattice as `site`).
        Returns -1 if absent. Same convention as `bonds`.
        """
        i, j, b = self.site_coords[site]
        if b_to is None:
            b_to = b
        c1 = j + delta[0] + self.basis_frac[b_to, 0]
        c2 = i + delta[1] + self.basis_frac[b_to, 1]
        return self.retrieve_index(c1, c2)

    def bond_arrays(self, deltas, b_from=0, b_to=None):
        """
        Concatenate several cell displacements into flat jnp int arrays (src, dst),
        ready to feed a jit-ed local energy. All displacements share b_from/b_to.
        """
        s_all, t_all = [], []
        for d in deltas:
            s, t = self.bonds(d, b_from=b_from, b_to=b_to)
            s_all.append(s)
            t_all.append(t)
        return (jnp.asarray(np.concatenate(s_all)),
                jnp.asarray(np.concatenate(t_all)))

    def shells(self, n_shells=None):
        """
        Distance-shell neighbour lists (for correlation functions / structure
        factors). Returns [(distance, src[K], dst[K]), ...] with each ordered
        pair counted once. NOT for Hamiltonian construction — use bonds() there.
        """
        dists = np.unique(self.dist_matrix)
        dists = dists[dists > 0]
        if n_shells is not None:
            dists = dists[:n_shells]
        out = []
        for d in dists:
            src, dst = np.nonzero(np.triu(self.dist_matrix == d, k=1))
            out.append((float(d), src, dst))
        return out

    def plot(self, filename=None):
        """Scatter the sites, coloured by sublattice and labelled by index."""
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 6))
        for b in range(self.nb):
            m = self.site_coords[:, 2] == b
            plt.scatter(self.points[m, 0], self.points[m, 1], s=40, label=f"basis {b}")
        for idx, p in enumerate(self.points):
            plt.text(p[0] + 0.05, p[1] + 0.05, str(idx), fontsize=9)
        plt.gca().set_aspect("equal")
        if self.nb > 1:
            plt.legend()
        if filename:
            plt.savefig(filename, dpi=150, bbox_inches="tight")
        else:
            plt.show()