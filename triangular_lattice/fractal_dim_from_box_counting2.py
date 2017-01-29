#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2017-01-29


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def read_from_csv(fn):
    data = np.loadtxt(fn, delimiter=',', skiprows=1)
    return data


if __name__ == '__main__':
    frames_list = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
    ##             0    1    2    3    4     5     6     7     8     9
    beta_list = [0, 2, 4, 6, 8, 10]
    ##           0  1  2  3  4  5

    Ds = read_from_csv('./results/img/fractal_dim/2017-01-27/data_170129.csv').T

    # Ds = manual_data()

    markers = ['o', 'v', '^', 's', 'D', 'h']

    fig, ax = plt.subplots()
    for i, beta in enumerate(beta_list):
        color = cm.viridis(float(i) / (len(beta_list) - 1))
        ax.plot(frames_list, Ds[i], marker=markers[i % len(markers)],
                ls='', color=color, label=r'$\beta = %2.2f$' % beta)
    # ax.legend(loc='best')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    fig.subplots_adjust(right=0.8)
    ax.set_title(r'Fractal dimension $D$')
    ax.set_xlabel(r'$T$')
    ax.set_ylabel(r'$D$')
    ax.set_xlim(0, 2200)
    ax.set_ylim(0.8, 2.)
    plt.show()
