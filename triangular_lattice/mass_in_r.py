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


class Count_in_r(Main):
    def __init__(self):
        L = 1000
        Main.__init__(self, Lx=L, Ly=L, plot=True,
                      frames=500,
                      beta=0.,
                      strings=[{'id': 1, 'x': L/4, 'y': L/2, 'vec': [0, 4]}])

def count_point_in_r(self, s, N_r):
    N = float(len(s.vec) + 1)
    pos = list(s.pos.T)
    x = self.lattice_X[pos]
    y = self.lattice_Y[pos]
    X = np.average(x)
    Y = np.average(y)
    R = np.sqrt(np.sum((x - X) ** 2 + (y - Y) ** 2) / N)
    dist = np.sqrt((x - X) ** 2 + (y - Y) ** 2)
    r = np.logspace(1, int(np.log2(max(dist))), num=N_r, base=2.)
    res = []
    for _r in r:
        res.append(len(np.where(dist < _r)[0]))
    return r, np.array(res)


def main():
    N_r = 100
    fig, ax = plt.subplots()

    main = Count_in_r()
    r, M = count_point_in_r(main, main.strings[0], N_r)

    index_stop = len(r) - 10
    ax.loglog(r, M)
    optimizer = Optimize_powerlaw(args=(r[:index_stop],
                                        M[:index_stop]),
                                    parameters=[0., 2.])
    result = optimizer.fitting()
    print "D = %f" % result['D']
    ax.loglog(r[:index_stop], optimizer.fitted(r[:index_stop]), lw=2,
              label='D = %f' % result['D'])

    ax.set_xlabel('Radius $r$ from the center of gravity')
    ax.set_ylabel('Mass in a circle with radius $r$')
    ax.set_title('$r$ vs. $M(r)$')
    ax.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    main()

