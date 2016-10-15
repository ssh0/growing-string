#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-10-15

import numpy as np
import matplotlib.pyplot as plt


def calc_tortuosity_for_each_beta(filename):
    data = np.load(filename)
    beta = data['beta']
    num_of_strings = data['num_of_strings']
    L = data['L']
    frames = data['frames']
    distance_list = data['distance_list']
    path_length = data['path_length']

    num_of_pairs = 300

    x = np.array(path_length)
    x = x.reshape((num_of_strings, int(x.shape[0] / num_of_strings / num_of_pairs), num_of_pairs))

    y = np.array(distance_list)
    y = y.reshape((num_of_strings, int(y.shape[0] / num_of_strings/ num_of_pairs), num_of_pairs))
    y = y / x

    y_ave = np.average(y, axis=2)

    L = x[0, :, 0]
    lambda_ave = np.average(y_ave, axis=0)
    return L, lambda_ave

if __name__ == '__main__':
    result_data_path = {
        '0': "./results/data/distances/beta=0.00_161012_171430.npz",
        '5': "./results/data/distances/beta=5.00_161012_171649.npz",
        '10': "./results/data/distances/beta=10.00_161012_172119.npz",
        '15': "./results/data/distances/beta=15.00_161012_172209.npz",
        '20': "./results/data/distances/beta=20.00_161012_172338.npz",
        '0-2': "./results/data/distances/beta=0.00_161015_153311.npz",
        '5-2': "./results/data/distances/beta=5.00_161015_153838.npz",
        '10-2': "./results/data/distances/beta=10.00_161015_154048.npz",
        '15-2': "./results/data/distances/beta=15.00_161015_154136.npz",
        '20-2': "./results/data/distances/beta=20.00_161015_154419.npz"
    }
    fig, ax = plt.subplots()
    for i in [0, 5, 10, 15, 20]:
        fn = result_data_path['%d-2' % i]
        ax.plot(*calc_tortuosity_for_each_beta(fn),
                ls='', marker='.', label=r'$\beta = %2.2f$' % float(i))
    ax.set_xlabel(r'Path length $L$')
    ax.set_ylabel(r'Tortuosity $T$')
    ax.set_title('Tortuosity $T = \lambda_{\mathrm{avg}} / L$')
    ax.legend(loc='best')
    plt.show()

