#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-08-19


from growing_string import Main
from Optimize import Optimize_powerlaw
import matplotlib.pyplot as plt
import numpy as np


class Roughness(Main):
    def __init__(self, L=60, frames=1000):
        Main.__init__(self, Lx=L, Ly=L, size=[3,] * 1, plot=False,
                      frames=frames,
                      strings=[{'id': 1, 'x': L/2, 'y': L/4, 'vec': [0, 4]}]
                      )

def get_surface_points(self, s):
    surface_points = []
    for i, (x, y) in enumerate(s.pos):
        nnx, nny = self.lattice.neighborhoods[x, y]
        for r in [0, 1, 2, 3, 4, 5]:
            nx, ny = nnx[r], nny[r]
            if nx == -1 or ny == -1:
                continue
            elif self.occupied[nx, ny]:
                continue
            else:
                surface_points.append(i)
    pos_index = list(set(surface_points))
    return list(np.array(s.pos[pos_index]).T)

def eval_fluctuation_on_surface(self, s):
    pos = get_surface_points(self, s)
    X = np.average(self.lattice_X[pos])
    Y = np.average(self.lattice_Y[pos])
    x = self.lattice_X[pos] - X
    y = self.lattice_Y[pos] - Y
    r = np.sqrt(x ** 2 + y ** 2)
    R = np.sqrt(np.average(x ** 2 + y ** 2))
    theta = np.arctan(y / x)
    theta[x < 0] = theta[x < 0] + np.pi
    theta = theta % (2 * np.pi)
    return np.array([theta, r]), R

def plot_to_veirfy(theta, r, R_t):
    fig = plt.figure()
    ax_left = fig.add_subplot(121)
    ax_right = fig.add_subplot(122)

    for theta_, r_ in zip(theta, r):
        x1, y1 = R_t * np.cos(theta_), R_t * np.sin(theta_)
        x2, y2 = r_ * np.cos(theta_), r_ * np.sin(theta_)
        ax_left.plot([x1, x2], [y1, y2], color='k', alpha=0.5)

    ax_left.plot(r * np.cos(theta), r * np.sin(theta), 'o')
    th = np.linspace(0., 2 * np.pi, num=100)
    ax_left.plot(R_t * np.cos(th), R_t * np.sin(th))
    ax_left.set_aspect('equal')
    ax_left.set_title('Real space')

    ax_right.plot(R_t * theta, r - R_t)
    ax_right.plot([0., 2 * np.pi * R_t], [0., 0.])
    ax_right.set_title('Fluctuation of surface')
    ax_right.set_xlabel(r'$ R \theta$')
    ax_right.set_ylabel(r'$r_{i} - R$')

    ax_left.set_title('Real space')

def eval_std_various_width(theta, r, R_t):
    L = theta * R_t
    L_max = 2. * np.pi * R_t

    res_width = []
    res_std = []
    width_sample = 50
    samples_N = 100

    log_width_min = np.log2(L_max / len(L)) + 1.
    log_width_max = np.log2(L_max)
    for width in np.logspace(log_width_min, log_width_max,
                             base=2., num=width_sample):
        stds = []
        for samples_start in np.linspace(0., L_max - width, num=samples_N):
            try:
                index_start = np.min(np.where(L > samples_start)[0])
                index_end = np.max(np.where(L < samples_start + width)[0])
            except ValueError:
                continue

            # if there are no points in
            if index_start > index_end:
                continue

            stds.append(np.std(r[index_start:index_end + 1]))

        if len(stds) == 0:
            continue
        else:
            res_width.append(width)
            res_std.append(np.mean(stds))

    return res_width, res_std

def plot_result(x, y):
    fig, ax = plt.subplots()
    ax.loglog(x, y, 'o-')
    ax.set_title('Roughness (averaged) at some width')
    ax.set_xlabel(r'width')
    ax.set_ylabel(r'$\sigma$')

def fitting(x, y, index_start, index_end):
    optimizer = Optimize_powerlaw(args=(x[index_start:index_end],
                                        y[index_start:index_end]),
                                  parameters=[0., 0.5])
    result = optimizer.fitting()
    print result['D']
    ax.loglog(x[index_start:index_end],
              optimizer.fitted(x[index_start:index_end]),
              lw=2, label='D = %f' % result['D'])
    ax.legend(loc='best')


if __name__ == '__main__':

    main = Roughness(L=120, frames=3000)
    (theta, r), R_t = eval_fluctuation_on_surface(main, main.strings[0])
    index_sorted = np.argsort(theta)
    theta, r = theta[index_sorted], r[index_sorted]
    # plot_to_veirfy(theta, r, R_t)

    res_width, res_std = eval_std_various_width(theta, r, R_t)
    plot_result(res_width, res_std)
    fitting(res_width, res_std, 5, 33)

    plt.show()
