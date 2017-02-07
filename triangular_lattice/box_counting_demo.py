#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2017-02-01

from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from growing_string import Main


def plot_boxes(self, delta):
    L = 100
    num = int(L / delta) + 1
    self.plot_all()

    xticks = np.linspace(0, self.lattice.Lx, num=num)
    yticks = np.linspace(0, self.lattice.Ly, num=num)
    self.ax.set_xticks(xticks)
    self.ax.set_yticks(yticks)
    self.ax.grid(zorder=0, ls='-')

    pos = list(self.strings[0].pos.T)
    x, y = self.lattice_X[pos], self.lattice_Y[pos]
    ps = []
    for x0, x1 in zip(xticks[:-1], xticks[1:]):
        for y0, y1 in zip(yticks[:-1], yticks[1:]):
            if np.any((x >= x0) & (y >= y0) & (x < x1) & (y < y1)):
                ps.append(patches.Rectangle((x0, y0),
                                            x1 - x0, y1 - y0,
                                            alpha=0.2, facecolor='k',
                                            edgecolor='none'))
    N = len(ps)
    print('N = {}'.format(N))
    for p in ps:
        self.ax.add_patch(p)
    self.ax.grid(zorder=0, ls='-')
    # self.ax.text(50, 80, r'$N(\delta = {}) = {}$'.format(delta, N), fontsize=36, ha='center')
    plt.show()


if __name__ == '__main__':
    L = 100

    main= Main(Lx=L, Ly=L, size=[3,] * 1, frames=1000,
               beta=3.,
               plot=True, plot_surface=False,
               interval=0,
               strings=[{'id': 1, 'x': L/4, 'y': L/2, 'vec': [0, 4]}]
               # strings=[{'id': 1, 'x': L/4, 'y': L/2, 'vec': [0, 4, 2]}]
               )


    #### box_counting_demo
    ## L = 100
    # deltas = np.array([1., 2., 4., 5., 10., 20.])
    deltas = np.array([2., 5., 10.])
    for delta in deltas:
        plot_boxes(main, delta)


