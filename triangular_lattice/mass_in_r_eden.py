#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-09-20

from eden import Eden
from Optimize import Optimize_powerlaw
import matplotlib.pyplot as plt
import numpy as np


class Count_in_r(Eden):
    def __init__(self):
        L = 60
        Eden.__init__(self, Lx=L, Ly=L, plot=False, frames=1000)

def count_point_in_r(self, N_r):
    N = float(len(self.points))
    pos = list(np.array(self.points).T)
    x = self.lattice_X[pos]
    y = self.lattice_Y[pos]
    X = np.average(x)
    Y = np.average(y)
    R = np.sqrt(np.sum((x - X) ** 2 + (y - Y) ** 2) / N)
    r = np.logspace(2, int(np.log2(R)) + 1, num=N_r, base=2.)
    dist = np.sqrt((x - X) ** 2 + (y - Y) ** 2)
    res = []
    for _r in r:
        res.append(len(np.where(dist < _r)[0]))
    return r, np.array(res)


if __name__ == '__main__':

    N_r = 100
    fig, ax = plt.subplots()

    main = Count_in_r()
    main.execute()
    r, res = count_point_in_r(main, N_r)

    ax.loglog(r, res)
    optimizer = Optimize_powerlaw(args=(r[:-20], res[:-20]), parameters=[0., 2.])
    result = optimizer.fitting()
    print "D = %f" % result['D']
    ax.loglog(r[:-20], optimizer.fitted(r[:-20]), lw=2,
              label='D = %f' % result['D'])

    ax.set_xlabel('Radius $r$ from the center of gravity')
    ax.set_ylabel('Mass in a circle with radius $r$')
    ax.set_title('$r$ vs. $M(r)$ on Eden model')
    ax.legend(loc='best')
    plt.show()
