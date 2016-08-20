#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-08-19


from growing_string import Main
from Optimize import Optimize_powerlaw
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class Roughness(Main):
    def __init__(self):
        L = 60
        Main.__init__(self, Lx=L, Ly=L, size=[3,] * 1, plot=False,
                      frames=1000,
                      strings=[{'id': 1, 'x': L/2, 'y': L/4, 'vec': [0, 4]}],
                      post_function=calc_roughness)

def calc_roughness(self, i, s):
    N = float(len(s.vec) + 1)
    pos = list(np.array(self.neighbors_set.keys()).T)
    X = np.sum(self.lattice_X[pos]) / N
    Y = np.sum(self.lattice_Y[pos]) / N
    R = np.sqrt(np.sum((self.lattice_X[pos] - X) ** 2
                       + (self.lattice_Y[pos] - Y) ** 2) / N)
    r = np.sqrt((self.lattice_X[pos] - X) ** 2
                + (self.lattice_Y[pos] - Y) ** 2)
    # Ra = np.sum(np.abs(r - R)) / (2 * np.pi * R)
    Ra = np.sum(np.abs(r - R)) / N
    return Ra


if __name__ == '__main__':

    fig, ax = plt.subplots()

    roughness = []
    num_strings = 10
    for s in tqdm(range(num_strings)):
        main = Roughness()
        roughness.append(main.post_func_res)

    # main = Roughness()
    # roughness = [main.post_func_res]

    steps = range(len(roughness[0]))
    # ax.loglog(steps, roughness[0])

    # # plot for all strings
    for s in range(num_strings):
        ax.loglog(steps, roughness[s], alpha=0.5)
    ax.loglog(steps, np.average(np.array(roughness), axis=0))

    # optimizer = Optimize_powerlaw(args=(steps[10:], roughness[0][10:]),
    #                               parameters=[0., 0.5])
    # result = optimizer.fitting()
    # ax.loglog(steps[10:], optimizer.fitted(steps[10:]), lw=2,
    #           label='D = %f' % result['D'])
    ax.set_xlabel('Steps N')
    ax.set_ylabel('Roughness')
    ax.set_title('Roughness')
    # ax.legend(loc='best')
    plt.show()
