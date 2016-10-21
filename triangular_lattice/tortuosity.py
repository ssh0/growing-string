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
    # tortuosity = length_of_the_curve / distance_betwee_the_ends_of_it
    y = x / y

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
        '20-2': "./results/data/distances/beta=20.00_161015_154419.npz",
        "0-3": "./results/data/distances/beta=0.00_161018_160133.npz",
        "1-3": "./results/data/distances/beta=1.00_161018_161118.npz",
        "2-3": "./results/data/distances/beta=2.00_161018_162849.npz",
        "3-3": "./results/data/distances/beta=3.00_161018_164500.npz",
        "4-3": "./results/data/distances/beta=4.00_161018_170824.npz",
        "5-3": "./results/data/distances/beta=5.00_161018_172135.npz",
        "6-3": "./results/data/distances/beta=6.00_161018_173918.npz",
        "7-3": "./results/data/distances/beta=7.00_161018_175342.npz",
        "8-3": "./results/data/distances/beta=8.00_161018_180914.npz",
        "9-3": "./results/data/distances/beta=9.00_161018_182543.npz",
        "10-3": "./results/data/distances/beta=10.00_161018_184136.npz",
        "0-4": "./results/data/distances/beta=0.00_161018_202019.npz",
        "5-4": "./results/data/distances/beta=5.00_161018_202404.npz",
        "10-4": "./results/data/distances/beta=10.00_161018_202616.npz",
        "15-4": "./results/data/distances/beta=15.00_161018_202846.npz",
        "20-4": "./results/data/distances/beta=20.00_161018_203002.npz",
    }
    fig, ax = plt.subplots()
    # for i in [0, 5, 10, 15, 20]:
    for i in [0, 2, 4, 8, 10]:
        # fn = result_data_path['%d-2' % i]
        fn = result_data_path['%d-3' % i]
        # fn = result_data_path['%d-4' % i]
        ax.plot(*calc_tortuosity_for_each_beta(fn),
                ls='', marker='.', label=r'$\beta = %2.2f$' % float(i))
    ax.plot(ax.get_xlim(), [1., 1.], 'k-')
    ax.set_ylim(0., ax.get_ylim()[1])
    ax.set_xlabel(r'Path length $L$')
    ax.set_ylabel(r'Tortuosity $T$')
    ax.set_title('Tortuosity $T = L / \lambda_{\mathrm{avg}}$')
    ax.legend(loc='best')
    plt.show()

