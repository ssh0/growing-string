#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-05-22


from triangular import LatticeTriangular as LT
from base import Main as base
import numpy as np
import random


class Main(base):

    def __init__(self, Lx=40, Ly=40, N=4, size=[5, 4, 10, 12], plot=True):
        # Create triangular lattice with given parameters
        self.lattice = LT(np.zeros((Lx, Ly), dtype=np.int),
                          scale=10., boundary='periodic')

        self.occupied = np.zeros((Lx, Ly), dtype=np.bool)
        self.number_of_lines = sum(size)

        # Put the strings to the lattice
        self.strings = self.create_random_strings(N, size)

        self.plot = plot
        self.interval = 100


if __name__ == '__main__':
    N = 5
    # main = Main(Lx=100, Ly=100, N=N, size=[random.randint(4, 12)] * N)
    main = Main(Lx=40, Ly=40, N=N, size=[random.randint(4, 12)] * N)

    # Plot triangular-lattice points, string on it, and so on
    if main.plot:
        main.plot_all()
    else:
        while True:
            try:
                main.update()
            except StopIteration:
                break
