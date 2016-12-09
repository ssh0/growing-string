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
    try:
        size_dist_ave = data['size_dist_ave']
        return load_data_averaged(_path)
    except KeyError:
        pass

    num_of_strings = data['num_of_strings']
    frames = data['frames']
    Ls = data['Ls'].astype(np.float)
    # Ls = (3 * Ls * (Ls + 1) + 1)
    size_dist = data['size_dist']

    N0 = np.array([l[1] for l in size_dist], dtype=np.float) / num_of_strings
    n0 = N0[1:]
    S = np.array([np.sum(l) for l in size_dist], dtype=np.float) / num_of_strings
    n1 = (S[1:] - n0) * 2.

    N = []
    for l in size_dist:
        dot = np.dot(np.arange(len(l)), np.array(l).T)
        N.append(dot)
    # N = np.array([np.dot(np.arange(len(l)), np.array(l).T) for l in size_dist])
    N_all = 3. * Ls * (Ls + 1.) + 1
    N = np.array(N, dtype=np.float) / num_of_strings
    N_minus = N_all - N

    N_minus_rate = N_minus / N_all

    n_minus = N_minus[1:] - N_minus[:-1]

    n1_ave = n1 / np.sum(n1)

    n2 = (6 * Ls[1:]) - (n0 + n1 + n_minus)

    return {
        'beta': beta,
        'num_of_strings': num_of_strings,
        'frames': frames,
        'Ls': Ls,
        'N_minus': N_minus,
        'N_minus_rate': N_minus_rate,
        'S': S,
        'n0': n0,
        'n1': n1,
        'n2': n2,
        'n_minus': n_minus,
        'n1_ave': n1_ave,
    }

def load_data_averaged(_path):
    data = np.load(_path)
    beta = data['beta']
    num_of_strings = data['num_of_strings']
    frames = data['frames']
    Ls = data['Ls'].astype(np.float)
    # Ls = (3 * Ls * (Ls + 1) + 1)
    # size_dist = data['size_dist']
    size_dist_ave = data['size_dist_ave']

    N0 = np.array([l[1] for l in size_dist_ave], dtype=np.float)
    n0 = N0[1:]
    S = np.array([np.sum(l) for l in size_dist_ave], dtype=np.float)
    n1 = (S[1:] - n0) * 2.

    N = []
    for l in size_dist_ave:
        dot = np.dot(np.arange(len(l)), np.array(l).T)
        N.append(dot)
    # N = np.array([np.dot(np.arange(len(l)), np.array(l).T) for l in size_dist_ave])
    N_all = 3. * Ls * (Ls + 1.) + 1
    N = np.array(N, dtype=np.float)
    N_minus = N_all - N

    N_minus_rate = N_minus / N_all

    n_minus = N_minus[1:] - N_minus[:-1]

    n1_ave = n1 / np.sum(n1)

    n2 = (6 * Ls[1:]) - (n0 + n1 + n_minus)

    return {
        'beta': beta,
        'num_of_strings': num_of_strings,
        'frames': frames,
        'Ls': Ls,
        'N_minus': N_minus,
        'N_minus_rate': N_minus_rate,
        'S': S,
        'n0': n0,
        'n1': n1,
        'n2': n2,
        'n_minus': n_minus,
        'n1_ave': n1_ave,
    }


def result_n2(path):
    fig, ax = plt.subplots()
    for i, result_data_path in enumerate(path):
        globals().update(load_data(result_data_path))
        ## datas
        ## n0: occupied, 0
        ## n1: occupied, 1
        ## n2: occupied, 2 (locked)
        ## n_minus: not occupied

        # ax.plot(Ls[1:], N_minus_rate[1:], '.', label=r'$\beta = %2.2f$' % beta,
        #         color=cm.viridis(float(i) / len(path)))

        # ax.plot(Ls[1:], n0, '.', label=r'$\beta = %2.2f$' % beta,
        #         color=cm.viridis(float(i) / len(path)))

        # ax.plot(Ls[1:], n0 / (6 * Ls[1:]), '.', label=r'$\beta = %2.2f$' % beta,
        #         color=cm.viridis(float(i) / len(path)))

        # ax.plot(Ls[1:], n1, '.', label=r'$\beta = %2.2f$' % beta,
        #         color=cm.viridis(float(i) / len(path)))

        # ax.plot(Ls[1:], n1 / (6 * Ls[1:]), '.', label=r'$\beta = %2.2f$' % beta,
        #         color=cm.viridis(float(i) / len(path)))

        # ax.plot(Ls[1:], n_minus, '.', label=r'$\beta = %2.2f$' % beta,
        #         color=cm.viridis(float(i) / len(path)))

        # ax.plot(Ls[1:], n_minus / (6 * Ls[1:]), '.', label=r'$\beta = %2.2f$' % beta,
        #         color=cm.viridis(float(i) / len(path)))

        # ax.plot(Ls[1:], n2, '.', label=r'$\beta = %2.2f$' % beta,
        #         color=cm.viridis(float(i) / len(path)))

        # ax.plot(Ls[1:], n2 / (6 * Ls[1:]), '.', label=r'$\beta = %2.2f$' % beta,
        #         color=cm.viridis(float(i) / len(path)))

        ax.plot(Ls[1:], S[1:], '.', label=r'$\beta = %2.2f$' % beta,
                color=cm.viridis(float(i) / len(path)))

        # ax.plot(Ls[1:], n0 + 0.5 * n1, '.', label=r'$\beta = %2.2f$' % beta,
        #         color=cm.viridis(float(i) / len(path)))

    ax.legend(loc='best')
    # ax.set_ylim((0., ax.get_ylim()[1]))
    ax.set_title('Strings in hexagonal region' +
                ' (sample: {})'.format(num_of_strings))
    ax.set_xlabel(r'Cutting size $L$')
    ax.set_ylabel(r'$n_{2}$')

    plt.show()


if __name__ == '__main__':
    result_n2(set_data_path.data_path)
