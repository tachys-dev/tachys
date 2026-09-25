# Lattices

A `Lattice` describes the geometry of a finite two-dimensional cluster: a
Bravais lattice spanned by two primitive vectors $\mathbf{a}_1$ and
$\mathbf{a}_2$, a basis of `nb` sites in each unit cell, the number of cells
along each direction, and the boundary conditions. It is immutable and lives on
the host, as NumPy arrays. Hamiltonians and Monte Carlo moves read their bonds
from it, and every `State` carries it as static metadata.

## Built-in lattices

`tachys.lattice.lattice_database` provides the common geometries. Apart from
`chain`, each factory takes `shape=(Lx, Ly)`, the number of unit cells along
$\mathbf{a}_1$ and $\mathbf{a}_2$, and the flags `pbc_x` and `pbc_y` for
periodic boundaries along each direction, both `True` by default.

| Factory | Bravais lattice | `nb` |
|---|---|---|
| `chain(L, pbc=True)` | one-dimensional | 1 |
| `square(shape)` | square | 1 |
| `cylinder(shape)` | square, open along $\mathbf{a}_1$ | 1 |
| `triangular(shape)` | triangular | 1 |
| `honeycomb(shape)` | triangular | 2 |
| `kagome(shape)` | triangular | 3 |
| `shastry_sutherland(shape)` | square | 4 |
| `plaquette(shape)` | square | 4 |

```python
from tachys.lattice.lattice_database import square, kagome

lattice = square(shape=(6, 6))
lattice.Ns, lattice.nb, lattice.L, lattice.pbc  # (36, 1, (6, 6), (True, True))

strip = square(shape=(6, 6), pbc_x=False)  # open along a1, periodic along a2
kag = kagome(shape=(3, 3))                 # 3 × 3 cells × 3 sites = 27 sites
```

## Custom lattices

Any other geometry is built with `Lattice.create`, from the two primitive
vectors, the Cartesian positions of the basis sites in one cell, and the shape.
The Lieb lattice, for example, is a square Bravais lattice with one site at each
corner of the unit cell and one in the middle of each edge:

```python
from tachys.lattice.lattice import Lattice

lieb = Lattice.create(
    a1=[1.0, 0.0],
    a2=[0.0, 1.0],
    basis=[[0.0, 0.0],      # b = 0: corner
           [0.5, 0.0],      # b = 1: middle of the horizontal edge
           [0.0, 0.5]],     # b = 2: middle of the vertical edge
    shape=(4, 4),
    pbc_x=True,
    pbc_y=True,
)
lieb.Ns                     # 48 = 4 × 4 cells × 3 sites
```

The built-in factories are defined the same way.

## Site indices

Sites are numbered cell by cell: row $i$ (along $\mathbf{a}_2$) outermost,
column $j$ (along $\mathbf{a}_1$) next, and sublattice $b$ innermost,

$$
s = (i\,L_x + j)\,n_b + b .
$$

Three arrays translate between the representations of a site:

- `lattice.site_coords[s]` is the triple `(i, j, b)`;
- `lattice.cell_to_site[i, j, b]` is the index `s`;
- `lattice.points[s]` is the Cartesian position.

`lattice.plot()` draws the sites, coloured by sublattice and labelled by index —
the quickest check that a custom lattice is the intended one.

## Bonds

Hamiltonians are built from bonds, and `lattice.bonds` generates them from a
displacement between unit cells. On a $3 \times 3$ square lattice, where sites
0, 1 and 2 form the first row:

```python
from tachys.lattice.lattice_database import square

lattice = square(shape=(3, 3))
src, dst = lattice.bonds((1, 0))      # every site and its neighbour along a1
src                                   # [0, 1, 2, 3, 4, 5, 6, 7, 8]
dst                                   # [1, 2, 0, 4, 5, 3, 7, 8, 6]
```

The bond from site 2 to site 0 crosses the periodic boundary.
`lattice.bonds(delta, b_from=0, b_to=None)` connects sublattice `b_from` in
every cell $C$ to sublattice `b_to` (by default `b_from`) in the cell
$C + \delta$, where $\delta = (d_1, d_2)$ counts whole cells along
$\mathbf{a}_1$ and $\mathbf{a}_2$. It returns two index arrays with one entry
per bond. With open boundaries, bonds that would leave the cluster are dropped.

Each call lists every bond of one type once, in one direction: `(1, 0)` and
`(-1, 0)` describe the same bonds, so only one of them belongs in a Hamiltonian.
On the square lattice, the nearest-neighbour bonds are `(1, 0)` and `(0, 1)`,
and the next-nearest (diagonal) ones `(1, 1)` and `(1, -1)`.

On a lattice with a basis, the offset inside the cell comes from the sublattice
indices, not from $\delta$. Each corner of the Lieb lattice has four neighbours:
the two edge sites of its own cell, and one edge site in each of the cells to
its left and below.

```python
from tachys.lattice.lattice import Lattice

lieb = Lattice.create(a1=[1.0, 0.0], a2=[0.0, 1.0], shape=(4, 4),
                      basis=[[0.0, 0.0],      # b = 0: corner
                             [0.5, 0.0],      # b = 1: middle of the horizontal edge
                             [0.0, 0.5]])     # b = 2: middle of the vertical edge

lieb_bonds = [
    ((0, 0), 0, 1), ((-1, 0), 0, 1),     # corner – horizontal-edge site
    ((0, 0), 0, 2), ((0, -1), 0, 2),     # corner – vertical-edge site
]
for delta, b_from, b_to in lieb_bonds:
    src, dst = lieb.bonds(delta, b_from=b_from, b_to=b_to)
```

The same displacements and sublattice indices are what the Hamiltonian
factories take, see {doc}`hamiltonians`.

## Distance shells

`lattice.shells(n)` groups the pairs of sites by distance and returns the `n`
closest shells as `(distance, src, dst)` tuples, each pair counted once. It is
a useful check on a list of bonds — the four displacements above must produce
exactly the first shell, four bonds for each of the 16 corners:

```python
from tachys.lattice.lattice import Lattice

lieb = Lattice.create(a1=[1.0, 0.0], a2=[0.0, 1.0], shape=(4, 4),
                      basis=[[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]])
distance, src, dst = lieb.shells(1)[0]    # distance 0.5, 64 pairs
```

`BondExchange` draws the pairs of sites it exchanges from these shells, see
{doc}`sampling`.
