#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-09-23
"""Eden model on triangular lattice"""

from eden import Eden
import numpy as np


def print_debug(arg):
    """Print argument if needed.

    You can use this function in any parts and its behavior is toggled here.
    """
    # print arg
    pass


if __name__ == '__main__':
    Lx, Ly = 60, 200
    eden = Eden(Lx, Ly, frames=5000, boundary={'h': 'periodic', 'v': 'reflective'})
    # eden = Eden(plot=False)
    eden.points = [(0, i) for i in range(Ly)]
    eden.occupied[list(np.array(eden.points).T)] = True
    eden.neighbors = [(1, i) for i in range(Ly)]

    eden.execute()

    print_debug(eden.occupied)
    print_debug(len(eden.neighbors))
    print_debug(len(np.where(eden.occupied)[0]))
    print_debug(len(eden.points))
    print_debug(eden.points)

