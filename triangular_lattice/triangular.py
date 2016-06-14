#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-05-12

import numpy as np


class LatticeTriangular(object):

    def __init__(self, lattice=None, boundary='periodic', scale=10, x0=0, y0=0):
        """Initialize triangular lattice

        --- Arguments ---
        boundary: Boundary condition ('periodic' or 'reflective')
        lattice (ndarray): Initial lattice condition (if given (ndarray of int))
        """
        if lattice is not None and type(lattice) is not np.ndarray:
            raise UserWarning("'lattice' must be np.ndarray \
                              (type(lattice) = %s)." % str(type(lattice)))

        self.Lx, self.Ly = lattice.shape

        if self.Lx % 2:
            raise UserWarning("Expected even row number for 'lattice' \
                              (given: %d)." % self.Lx)
        if boundary not in ('periodic', 'reflective'):
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

    def neighbor_of(self, i, j):

        def get_for_i_periodic(i):
            xu = (i - 1) % self.Lx
            xd = (i + 1) % self.Lx
            return xu, xd

        def get_for_i_reflective(i):
            if i == 0:
                xu = None
                xd = 1
            elif i == self.Lx - 1:
                xu = self.Lx - 2
                xd = None
            else:
                xu = i - 1
                xd = i + 1
            return xu, xd

        def get_for_j_periodic(j):
            yl = (j - 1) % self.Ly
            yr = (j + 1) % self.Ly
            return yl, yr

        def get_for_j_reflective(j):
            if j == int(i / 2):
                yl = None
                yr = (j + 1) % self.Ly
            elif j == (self.Ly - 1 + int(i / 2)) % self.Ly:
                yl = j - 1
                yr = None
            else:
                yl = (j - 1) % self.Ly
                yr = (j + 1) % self.Ly
            return yl, yr

        if self.boundary == 'periodic':
            xu, xd = get_for_i_periodic(i)
            yl, yr = get_for_j_periodic(j)
            neighbors = (np.array([i, xu, xu, i, xd, xd], dtype=np.int),
                         np.array([yr, yr, j, yl, yl, j], dtype=np.int))
        elif self.boundary == 'reflective':
            # something wrong in this part TODO : to be fixed
            xu, xd = get_for_i_reflective(i)
            yl, yr = get_for_j_reflective(j)
            neighbors_x = [i, xu, xu, i, xd, xd]
            neighbors_y = [yr, yr, j, yl, yl, j]
            neighbors = (neighbors_x, neighbors_y)
        return neighbors

    def to_realspace(self):
        dx = self.scale / self.Ly
        dy = self.scale / self.Lx
        unit_lengh = min(dx, (2 / np.sqrt(3)) * dy)
        self.dx = unit_lengh
        self.dy = unit_lengh * (np.sqrt(3) / 2)
        positions = np.array([])
        X = [((0.5 * i + j) * self.dx) % (self.dx * self.Ly) + self.x0
             for i in range(self.Lx) for j in range(self.Ly)]
        Y = [(0.5 + i) * self.dy + self.y0
             for i in range(self.Lx) for j in range(self.Ly)]
        return np.array(X), np.array(Y)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.tri as tri

    Lx, Ly = 4, 5
    lattice = np.random.randint(0, 6, size=(Lx, Ly))
    trilattice = LatticeTriangular(lattice, boundary='periodic')

    # print trilattice.neighborhoods[0, 0]

    X, Y = trilattice.to_realspace()
    triang = tri.Triangulation(X, Y)

    plt.figure()
    plt.triplot(triang, 'bo-')

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
