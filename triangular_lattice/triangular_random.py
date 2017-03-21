#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2017-03-03

import numpy as np


class LatticeTriangularRandom(object):
    """論文[1]に従い六角格子内の1点をその六角格子セルを代表する点とみなし，
    隣接する6つのセルを代表する点を繋ぐことでランダムな三角格子を生成する
    [1] https://www.jstage.jst.go.jp/article/journalcpij/44.3/0/44.3_799/_pdf
    """

    def __init__(self, lattice=None, boundary={'h': 'periodic', 'v': 'periodic'},
                 scale=10., x0=0, y0=0
                 ):
        """Initialize triangular lattice

        --- Arguments ---
        boundary: Boundary conditions
                  {h: <condition>,  # horizontal boundary condition
                   v: <condition>   # vertical boundary condition
                  }
                  condition : ('periodic' or 'reflective')
        lattice (ndarray): Initial lattice condition (if given (ndarray of int))
        """
        if lattice is not None and type(lattice) is not np.ndarray:
            raise UserWarning("'lattice' must be np.ndarray \
                              (type(lattice) = %s)." % str(type(lattice)))

        self.Lx, self.Ly = lattice.shape

        if self.Lx % 2:
            raise UserWarning("Expected even row number for 'lattice' \
                              (given: %d)." % self.Lx)
        for condition in boundary.values():
            if condition not in ('periodic', 'reflective'):
                raise UserWarning("'boundary' must be 'periodic' or 'reflective' \
                                (given: boudary = %s)." % boundary)

        self.boundary = boundary
        self.scale = float(scale)
        self.x0, self.y0 = x0, y0

        if self.boundary['h'] == 'periodic' and self.boundary['v'] == 'periodic':
            self.neighbor_of = self.neighbor_of_periodic
        elif self.boundary['h'] == 'reflective' and self.boundary['v'] == 'reflective':
            self.neighbor_of = self.neighbor_of_reflective
        else:
            self.neighbor_of = self.neighbor_of_combo

        self.coordinates_x, self.coordinates_y = self.to_realspace()

    def get_for_i_periodic(self, i):
        xu = (i - 1) % self.Lx
        xd = (i + 1) % self.Lx
        return xu, xd

    def get_for_i_reflective(self, i):
        if i == 0:
            xu = -1
            xd = 1
        elif i == self.Lx - 1:
            xu = i - 1
            xd = -1
        else:
            xu = i - 1
            xd = i + 1
        return xu, xd

    def get_for_j_periodic(self, j):
        yl = (j - 1) % self.Ly
        yr = (j + 1) % self.Ly
        return yl, yr

    def get_for_j_reflective(self, j):
        if j == 0:
            yl = -1
            yr = 1
        elif j == self.Ly - 1:
            yl = j - 1
            yr = -1
        else:
            yl = j - 1
            yr = j + 1
        return yl, yr

    def neighbor_of_periodic(self, i, j):
        xu = (i - 1) % self.Lx
        xd = (i + 1) % self.Lx
        yl = (j - 1) % self.Ly
        yr = (j + 1) % self.Ly
        return [xd, i, xu, xu, i, xd], [j, yr, yr, j, yl, yl]

    def neighbor_of_reflective(self, i, j):
        if i == 0:
            xu = -1
            xd = 1
        elif i == self.Lx - 1:
            xu = i - 1
            xd = -1
        else:
            xu = i - 1
            xd = i + 1
        if j == 0:
            yl = -1
            yr = 1
        elif j == self.Ly - 1:
            yl = j - 1
            yr = -1
        else:
            yl = j - 1
            yr = j + 1
        return [xd, i, xu, xu, i, xd], [j, yr, yr, j, yl, yl]

    def neighbor_of_combo(self, i, j):
        xu, xd = getattr(self, 'get_for_i_' + self.boundary['h'])(i)
        yl, yr = getattr(self, 'get_for_j_' + self.boundary['v'])(j)
        neighbors_x = np.array([xd, i, xu, xu, i, xd], dtype=np.int)
        neighbors_y = np.array([j, yr, yr, j, yl, yl], dtype=np.int)
        return neighbors_x, neighbors_y


    def to_realspace(self):
        unit_lengh = min(self.scale / self.Lx,
                         (2 / np.sqrt(3)) * (self.scale / self.Ly))
        self.dx, self.dy = unit_lengh, unit_lengh * (np.sqrt(3) / 2)
        # self.dx = 1, self.dy = sqrt(3) / 2
        y = np.arange(self.Ly)

        if self.boundary['h'] == 'periodic':
            X = ((2 * np.mgrid[:self.Ly, :self.Lx][1].T + y) % (2 * self.Lx)) \
                * (0.5 * self.dx) + self.x0
        elif self.boundary['h'] == 'reflective':
            X = (2 * np.mgrid[:self.Ly, :self.Lx][1].T + y) \
                * (0.5 * self.dx) + self.x0

        Y = (np.mgrid[:self.Lx, :self.Ly][1] + 0.5) * self.dy + self.y0
        X, Y = X.flatten(), Y.flatten()
        return X, Y


def pick_param():
    """Pick up random point from hex region"""
    # -1 <= p <= 1
    p = 2. * np.random.rand() - 1.
    # -1 <= q <= 1
    q = 2. * np.random.rand() - 1.
    if p+q >= -1 and p+q <= 1:
        return p, q
    else:
        return pick_param()

def randomize(lattice):
    X, Y = lattice.coordinates_x, lattice.coordinates_y

    ab = np.array([pick_param() for n in range(X.shape[0])])
    xy = np.dot(ab,
                np.array([
                    [1., 0.],
                    [0.5, np.sqrt(3)/2]
                ])
                )
    X, Y = X + 0.5 * lattice.dx * xy.T[0], Y + 0.5 * lattice.dx * xy.T[1]
    lattice.coordinates_x, lattice.coordinates_y = X, Y

    return X, Y

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.tri as tri


    fig, ax = plt.subplots()

    Lx, Ly = 50, 30
    lattice = LatticeTriangularRandom(
        np.zeros((Lx, Ly), dtype=np.int),
        scale=float(max(Lx, Ly)),
        boundary={'h': 'periodic',
                  'v': 'reflective'}
    )

    lattice_X = lattice.coordinates_x
    lattice_Y = lattice.coordinates_y
    X_min, X_max = min(lattice_X) - 0.7, max(lattice_X) + 0.7
    Y_min, Y_max = min(lattice_Y) - 0.5, max(lattice_Y) + 0.5
    ax.set_xlim([X_min, X_max])
    ax.set_ylim([Y_min, Y_max])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')

    _triang = tri.Triangulation(lattice_X, lattice_Y)
    ax.scatter(lattice_X, lattice_Y, s=5)

    randomize(lattice)

    triang = tri.Triangulation(lattice.coordinates_x, lattice.coordinates_y, triangles=_triang.triangles)
    # triang = tri.Triangulation(lattice.coordinates_x, lattice.coordinates_y)
    # ax.triplot(triang, color='#d5d5d5', lw=0.5)
    ax.triplot(triang, color='k', lw=0.5)


    # trilattice.lattice[neighbors] = 2
    # colorseq = np.zeros((Lx, Ly))
    # colorseq[trilattice.lattice == 2] = 0.9
    # colorseq[trilattice.lattice == 0] = 0.
    # colorseq[trilattice.lattice == 1] = 0.5

    # X, Y = trilattice.to_realspace(scale=20, x0=-10, y0=-10)
    # import matplotlib.pyplot as plt
    # plt.scatter(X, Y, s=100., c=colorseq)
    # plt.show()

    plt.show()
