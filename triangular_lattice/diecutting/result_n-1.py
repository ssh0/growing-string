#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-12-07

import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.cm as cm
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import gamma
import set_data_path


def load_data(_path):
    data = np.load(_path)
    beta = data['beta']
    num_of_strings = data['num_of_strings']
    frames = data['frames']
    Ls = data['Ls'].astype(np.float)
    # Ls = (3 * Ls * (Ls + 1) + 1)
    size_dist = data['size_dist']

    N0 = np.array([l[1] for l in size_dist])
    S = np.array([np.sum(l) for l in size_dist])
    N1 = (S - N0) * 2.
    N = []
    for l in size_dist:
        N.append(np.dot(np.arange(len(l)), np.array(l).T))
    N_all = 3 * Ls * (Ls + 1) + 1
    N = np.array(N, dtype=np.float) / num_of_strings
    N_minus = N_all - N
    N_rate = (N_all - N) / N_all
    n_minus = N_minus[1:] - N_minus[:-1]
    n_minus = np.array([1,] + n_minus.tolist())
    # n_minus = (np.array([1,] + n_minus.tolist())) / (6 * Ls)

    N1_ave = N1 / np.sum(N1)

    return {
        'beta': beta,
        'num_of_strings': num_of_strings,
        'frames': frames,
        'Ls': Ls,
        'N0': N0,
        'N1': N1,
        'N_rate': N_rate,
        'N_minus': N_minus,
        'n_minus': n_minus,
        'N1_ave': N1_ave,
        'S': S
    }

def result_n0(path):
    fig, ax = plt.subplots()
    for i, result_data_path in enumerate(path):
        globals().update(load_data(result_data_path))
        # ax.plot(Ls, S, '.', label=r'$\beta = %2.2f$' % beta,
        #         color=cm.viridis(float(i) / len(path)))
        # ax.plot(Ls, N0, '.', label=r'$\beta = %2.2f$' % beta,
        #         color=cm.viridis(float(i) / len(path)))
        # ax.plot(Ls, N1 / (6. * Ls) , '.', label=r'$\beta = %2.2f$' % beta,
        #         color=cm.viridis(float(i) / len(path)))
        # ax.plot(Ls, N_rate , '.', label=r'$\beta = %2.2f$' % beta,
        #         color=cm.viridis(float(i) / len(path)))
        ax.plot(Ls, n_minus , '.', label=r'$\beta = %2.2f$' % beta,
                color=cm.viridis(float(i) / len(path)))

    ax.legend(loc='best')
    # ax.set_ylim((0., 0.05))
    ax.set_title('Strings in hexagonal region' +
                ' (sample: {})'.format(num_of_strings))
    ax.set_xlabel(r'Cutting size $L$')
    ax.set_ylabel(r'$n_{-1}$')

    plt.show()


if __name__ == '__main__':
    result_n0(set_data_path.data_path)
