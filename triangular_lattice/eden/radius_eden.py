#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-09-20


import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eden import Eden
from optimize import Optimize_powerlaw
import matplotlib.pyplot as plt
import numpy as np


class Radius(Eden):
    def __init__(self):
        L = 120
        Eden.__init__(self, Lx=L, Ly=L, plot=False,
                      frames=3000,
                      post_function=calc_radius_of_rotation)

def calc_radius_of_rotation(self):
    # calc center
    N = float(len(self.points))
    pos = list(np.array(self.points).T)
    X = np.average(self.lattice_X[pos])
    Y = np.average(self.lattice_Y[pos])
    r = np.sqrt(np.sum((self.lattice_X[pos] - X) ** 2
                       + (self.lattice_Y[pos] - Y) ** 2) / N)
    return r


if __name__ == '__main__':

    fig, ax = plt.subplots()

    main = Radius()
    main.execute()
    radius_of_rotation = main.post_func_res

    steps = np.arange((len(radius_of_rotation)))
    ax.loglog(steps, radius_of_rotation)

    c_start = 100
    optimizer = Optimize_powerlaw(args=(steps[c_start:],
                                        radius_of_rotation[c_start:]),
                                  parameters=[0., 0.5])
    result = optimizer.fitting()
    ax.loglog(steps[c_start:], optimizer.fitted(steps[c_start:]), lw=2,
              label='D = %f' % result['D'])
    ax.set_xlabel('Steps N')
    ax.set_ylabel('Radius of rotation')
    ax.set_title('Raidus of rotation')
    ax.legend(loc='best')
    plt.show()
