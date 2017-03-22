#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2017-03-22
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from triangular import LatticeTriangular as LT
from triangular_random import randomize
from random_lattice.base_random_lattice import Main as base
import matplotlib.tri as tri
import numpy as np
import random


class Main(base):

    def __init__(self, Lx=40, Ly=40, N=4, size=[5, 4, 10, 12], plot=True):
        # Create triangular lattice with given parameters
        self.lattice = LT(np.zeros((Lx, Ly), dtype=np.int),
                          scale=10., boundary={'h': 'periodic', 'v': 'periodic'})
        randomize(self.lattice)
        self.triang_standard = tri.Triangulation(self.lattice.coordinates_x,
                                                 self.lattice.coordinates_y)
        self.triang_random = tri.Triangulation(
            self.lattice.coordinates_x,
            self.lattice.coordinates_y,
            triangles=self.triang_standard.triangles)

        self.occupied = np.zeros((Lx, Ly), dtype=np.bool)
        self.number_of_lines = sum(size)

        # Put the strings to the lattice
        self.strings = self.create_random_strings(N, size)

        self.plot = plot
        self.interval = 50
        self.frames = 1000


if __name__ == '__main__':
    N = 5
    # main = Main(Lx=100, Ly=100, N=N, size=[random.randint(4, 12)] * N)
    main = Main(Lx=40, Ly=40, N=N, size=[random.randint(8, 20)] * N)

    # Plot triangular-lattice points, string on it, and so on
    if main.plot:
        main.plot_all()
    else:
        while True:
            try:
                main.update()
            except StopIteration:
                break
