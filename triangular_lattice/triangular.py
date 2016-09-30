#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-05-12

import numpy as np


class LatticeTriangular(object):

    def __init__(self, lattice=None, boundary={'h': 'periodic', 'v': 'periodic'},
                 scale=10, x0=0, y0=0
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

        self.neighborhoods = []
        for i in range(self.Lx):
            tmp = []
            for j in range(self.Ly):
                tmp.append(self.neighbor_of(i, j)[:])
            self.neighborhoods.append(tmp)
        self.neighborhoods = np.array(self.neighborhoods)
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

    def neighbor_of(self, i, j):
        xu, xd = getattr(self, 'get_for_i_' + self.boundary['h'])(i)
        yl, yr = getattr(self, 'get_for_j_' + self.boundary['v'])(j)
        neighbors_x = np.array([xd, i, xu, xu, i, xd], dtype=np.int)
        neighbors_y = np.array([j, yr, yr, j, yl, yl], dtype=np.int)
        neighbors = (neighbors_x, neighbors_y)
        return neighbors

    def to_realspace(self):
        dx = self.scale / self.Lx
        dy = self.scale / self.Ly
        unit_lengh = min(dx, (2 / np.sqrt(3)) * dy)
        self.dx = unit_lengh
        self.dy = unit_lengh * (np.sqrt(3) / 2)
        # self.dx = 1, self.dy = sqrt(3) / 2

        if self.boundary['h'] == 'periodic':
            X = [((0.5 * j + i) * self.dx) % (self.dx * self.Lx) + self.x0
                 for i in range(self.Lx) for j in range(self.Ly)]
        elif self.boundary['h'] == 'reflective':
            X = [((0.5 * j + i) * self.dx) + self.x0
                 for i in range(self.Lx) for j in range(self.Ly)]

        Y = [(0.5 + j) * self.dy + self.y0
             for i in range(self.Lx) for j in range(self.Ly)]

        return np.array(X), np.array(Y)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.tri as tri

    fig, ax = plt.subplots()

    Lx, Ly = 50, 30
    lattice = LatticeTriangular(
        np.zeros((Lx, Ly), dtype=np.int),
        scale=float(max(Lx, Ly)),
        boundary={'h': 'periodic',
                  'v': 'reflective'}
    )

    lattice_X = lattice.coordinates_x
    lattice_Y = lattice.coordinates_y
    X_min, X_max = min(lattice_X) + 0.5, max(lattice_X) - 0.5
    Y_min, Y_max = min(lattice_Y) + 0.2, max(lattice_Y) - 0.2
    ax.set_xlim([X_min, X_max])
    ax.set_ylim([Y_min, Y_max])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')

    triang = tri.Triangulation(lattice_X, lattice_Y)
    ax.triplot(triang, color='#d5d5d5', marker='.', markersize=1)


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
