import numpy as np
from tachys.lattice.lattice import Lattice


def chain(L, pbc=True) -> Lattice:
    return Lattice.create(
        a1=[1.0, 0.0],
        a2=[0.0, 1.0],
        basis=[[0.0, 0.0]],
        shape=(L, 1),
        pbc_x=pbc,
        pbc_y=False,
    )


def square(shape, pbc_x=True, pbc_y=True) -> Lattice:
    return Lattice.create(
        a1=[1.0, 0.0],
        a2=[0.0, 1.0],
        basis=[[0.0, 0.0]],
        shape=shape,
        pbc_x=pbc_x,
        pbc_y=pbc_y,
    )


def triangular(shape, pbc_x=True, pbc_y=True) -> Lattice:
    alpha = np.pi / 3
    return Lattice.create(
        a1=[1.0, 0.0],
        a2=[np.cos(alpha), np.sin(alpha)],
        basis=[[0.0, 0.0]],
        shape=shape,
        pbc_x=pbc_x,
        pbc_y=pbc_y,
    )


def honeycomb(shape, pbc_x=True, pbc_y=True) -> Lattice:
    """Honeycomb lattice: the same triangular Bravais lattice as `triangular`,
    with a 2-site basis (A at the cell origin, B at the centroid (a1+a2)/3).
    Each A site has 3 NN B sites, reached by cell displacements (0,0), (-1,0),
    (0,-1) -- a Kitaev-model Hamiltonian assigns each of these its own
    bond-dependent operator (see e.g. tachys_projects/kitaev_honeycomb).
    """
    alpha = np.pi / 3
    a1 = np.array([1.0, 0.0])
    a2 = np.array([np.cos(alpha), np.sin(alpha)])
    t = (a1 + a2) / 3.0
    return Lattice.create(
        a1=a1,
        a2=a2,
        basis=[[0.0, 0.0], list(t)],
        shape=shape,
        pbc_x=pbc_x,
        pbc_y=pbc_y,
    )


def kagome(shape, pbc_x=True, pbc_y=True) -> Lattice:
    alpha = np.pi / 3
    return Lattice.create(
        a1=[1.0, 0.0],
        a2=[np.cos(alpha), np.sin(alpha)],
        basis=[[0.0, 0.0], [0.5, 0.0], [0.25, np.sqrt(3) / 4]],
        shape=shape,
        pbc_x=pbc_x,
        pbc_y=pbc_y,
    )


def cylinder(shape, pbc_x=False, pbc_y=True) -> Lattice:
    """Square lattice on a cylinder: PBC along y (rows) and OBC along x (columns) by default."""
    return Lattice.create(
        a1=[1.0, 0.0],
        a2=[0.0, 1.0],
        basis=[[0.0, 0.0]],
        shape=shape,
        pbc_x=pbc_x,
        pbc_y=pbc_y,
    )


def shastry_sutherland(shape, pbc_x=True, pbc_y=True) -> Lattice:
    alpha = 10.0 * np.pi / 180.0
    basis = [
        [0.0,                                                      0.0],
        [0.5 * np.tan(alpha),                                      0.5],
        [np.cos(0.25 * np.pi - alpha) / (np.cos(alpha) * np.sqrt(2.0)),
         np.sin(0.25 * np.pi - alpha) / (np.cos(alpha) * np.sqrt(2.0))],
        [0.5,                                                     -0.5 * np.tan(alpha)],
    ]
    return Lattice.create(
        a1=[1.0, 0.0],
        a2=[0.0, 1.0],
        basis=basis,
        shape=shape,
        pbc_x=pbc_x,
        pbc_y=pbc_y,
    )


def plaquette(shape, pbc_x=True, pbc_y=True) -> Lattice:
    """2x2-site plaquette lattice with near-square intra-cell geometry."""
    basis = [
        [0.0,  0.0],
        [0.45, 0.0],
        [0.0,  0.45],
        [0.45, 0.45],
    ]
    return Lattice.create(
        a1=[1.0, 0.0],
        a2=[0.0, 1.0],
        basis=basis,
        shape=shape,
        pbc_x=pbc_x,
        pbc_y=pbc_y,
    )
