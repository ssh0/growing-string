#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-08-15

from growing_string import Main
from Optimize import Optimize_powerlaw
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class Radius(Main):
    def __init__(self):
        L = 60
        Main.__init__(self, Lx=L, Ly=L, size=[3,] * 1, plot=False,
                      frames=1000,
                      strings=[{'id': 1, 'x': L/2, 'y': L/4, 'vec': [0, 4]}],
                      pre_function=calc_radius_of_rotation)

def calc_radius_of_rotation(self, i, s):
    # calc center
    N = float(len(s.vec) + 1)
    pos = list(s.pos.T)
    X = np.sum(self.lattice_X[pos]) / N
    Y = np.sum(self.lattice_Y[pos]) / N
    r = np.sqrt(np.sum((self.lattice_X[pos] - X) ** 2
                       + (self.lattice_Y[pos] - Y) ** 2) / N)
    return r


if __name__ == '__main__':

    fig, ax = plt.subplots()

    # radius_of_rotation = []
    # num_strings = 10
    # for s in tqdm(range(num_strings)):
    #     main = Radius()
    #     radius_of_rotation.append(main.pre_func_res)

    main = Radius()
    radius_of_rotation = [main.pre_func_res]

    steps = range(len(radius_of_rotation[0]))
    ax.loglog(steps, radius_of_rotation[0])

    # plot for all strings
    # for s in range(num_strings):
    #     ax.loglog(steps, radius_of_rotation[s], alpha=0.5)
    # ax.loglog(steps, np.average(np.array(radius_of_rotation), axis=0))

    optimizer = Optimize_powerlaw(args=(steps[10:], radius_of_rotation[0][10:]),
                                  parameters=[0., 0.5])
    result = optimizer.fitting()
    ax.loglog(steps[10:], optimizer.fitted(steps[10:]), lw=2,
              label='D = %f' % result['D'])
    ax.set_xlabel('Steps N')
    ax.set_ylabel('Radius of rotation')
    ax.set_title('Raidus of rotation')
    ax.legend(loc='best')
    plt.show()
