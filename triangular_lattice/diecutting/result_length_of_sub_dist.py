#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-12-06

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import set_data_path


def result_length_of_sub_dist(path):
    fig, ax = plt.subplots()
    for i, result_data_path in enumerate(path):
        data = np.load(result_data_path)
        beta = data['beta']
        num_of_strings = data['num_of_strings']
        L = data['L']
        frames = data['frames']
        Ls = data['Ls'].astype(np.float)
        size_dist = data['size_dist']
        N = np.sum(size_dist, axis=0)
        N_freq = N / np.sum(N)
        X = np.arange(1, len(N)+1)
        ax.semilogy(X, N_freq, '.', label=r'$\beta = %2.2f$' % beta,
                  color=cm.viridis(float(i) / len(path)))
        # ax.semilogy(X[::2], N_freq[::2], '.', label=r'$\beta = %2.2f$' % beta,
        #             color=cm.viridis(float(i) / len(path)))
        # ax.semilogy(X[1::2], N_freq[1::2], '.', label=r'$\beta = %2.2f$' % beta,
        #           color=cm.viridis(float(i) / len(path)))

    ax.legend(loc='best')
    ax.set_title("Appearance frequency of the subcluster of size $N$")
    ax.set_xlabel("$N$")
    ax.set_ylabel("Freq")
    plt.show()

if __name__ == '__main__':
    result_length_of_sub_dist(set_data_path.data_path)
