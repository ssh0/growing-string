#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-08-15

from growing_string import Main
from optimize import Optimize_powerlaw
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class Radius(Main):
    def __init__(self, params):
        Main.__init__(self, **params)

def calc_radius_of_rotation(self, i, s):
    # calc center
    N = float(len(s.vec) + 1)
    pos = list(s.pos.T)
    X = np.sum(self.lattice_X[pos]) / N
    Y = np.sum(self.lattice_Y[pos]) / N
    r = np.sqrt(np.sum((self.lattice_X[pos] - X) ** 2
                       + (self.lattice_Y[pos] - Y) ** 2) / N)
    return r

def main(beta=0., output=None, L=60, frames=1000, plot=False,
         save_image=False, filename_image="",
         plot_optimized=True):
    if output != None:
        save = True
    else:
        save = False

    # radius_of_rotation = []
    # num_strings = 10
    # for s in tqdm(range(num_strings)):
    #     main = Radius()
    #     radius_of_rotation.append(main.pre_func_res)

    params = {
        'Lx': L,
        'Ly': L,
        'frames': frames,
        'size': [3,] * 1,
        'plot': plot,
        'save_image': save_image,
        'filename_image': filename_image,
        'beta': beta,
        'strings': [{'id': 1, 'x': L/4, 'y': L/2, 'vec': [0, 4]}],
        'pre_function': calc_radius_of_rotation
    }

    main = Radius(params)
    radius_of_rotation = [main.pre_func_res]
    steps = range(len(radius_of_rotation[0]))

    # plot for all strings
    # for s in range(num_strings):
    #     ax.loglog(steps, radius_of_rotation[s], alpha=0.5)
    # ax.loglog(steps, np.average(np.array(radius_of_rotation), axis=0))

    index_start = 0
    optimizer = Optimize_powerlaw(args=(steps[index_start:],
                                        radius_of_rotation[0][index_start:]),
                                  parameters=[0., 0.5])
    result = optimizer.fitting()

    if plot_optimized or save:
        fig, ax = plt.subplots()
        ax.loglog(steps, radius_of_rotation[0])
        ax.loglog(steps[index_start:], optimizer.fitted(steps[index_start:]), lw=2,
                label='D = %f' % result['D'])
        ax.set_xlabel('Steps N')
        ax.set_ylabel('Radius of rotation')
        ax.set_title('Raidus of rotation')
        ax.legend(loc='best')
        if save:
            fig.savefig(output)
            plt.close()
        else:
            plt.show()
    else:
        return result['D']


if __name__ == '__main__':
    main()

