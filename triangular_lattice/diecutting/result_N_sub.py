#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-12-06

## for N_sub ===========

import set_data_path
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import gamma

def result_N_sub(path):
    fig, ax = plt.subplots()
    for result_data_path in path:
        data = np.load(result_data_path)
        beta = data['beta']
        num_of_strings = data['num_of_strings']
        L = data['L']
        frames = data['frames']
        Ls = data['Ls'].astype(np.float)
        N_sub = data['N_sub']

        # M = N_sub / 3 * Ls * (Ls + 1) + 1
        M = N_sub
        M_ave = M / np.sum(M)
        popt = curve_fit(gamma.pdf, xdata=Ls, ydata=M_ave, p0=[2.5, -5., 30])[0]
        print beta, popt
        ax.plot(Ls, M_ave, '.-', label=r'$\beta = %2.2f$' % beta)
        x = np.linspace(1., max(Ls), num=5*max(Ls))
        ax.plot(x, gamma.pdf(x, a=popt[0], loc=popt[1], scale=popt[2]),
                '-', label=r'fitted $\beta = %2.2f$' % beta)
    ax.legend(loc='best')
    ax.set_ylim((0., 0.1))
    ax.set_title('Strings in hexagonal region' +
                ' (sample: {})'.format(num_of_strings))
    ax.set_xlabel(r'Cutting size $L$')
    ax.set_ylabel('Average number of the sub-clusters in the hexagonal region of size L')

    plt.show()

if __name__ == '__main__':


    result_data_path_base = "../results/data/diecutting/"
    fn = [
        # "beta=0.00_161016_190044.npz",
        # "beta=0.10_161016_190056.npz",
        # "beta=0.20_161016_190105.npz",
        # "beta=0.30_161016_190111.npz",
        # "beta=0.40_161016_190125.npz",
        # "beta=0.50_161016_190158.npz",
        # "beta=0.60_161016_190207.npz",
        # "beta=0.70_161016_190217.npz",
        # "beta=0.80_161016_190229.npz",
        # "beta=0.90_161016_190236.npz",
        # "beta=1.00_161016_190242.npz",
        # "beta=2.00_161016_195423.npz",
        # "beta=3.00_161016_195439.npz",
        # "beta=4.00_161016_195447.npz",
        # "beta=5.00_161016_195452.npz",
        # "beta=6.00_161016_195526.npz",
        # "beta=0.00_161111_132832.npz",
        # "beta=0.00_161111_141810.npz",  # <- beta = 0.001
        "beta=1.00_161111_132834.npz",
        "beta=2.00_161111_132842.npz",
        "beta=3.00_161111_132849.npz",
        "beta=4.00_161111_132858.npz",
        "beta=5.00_161111_132907.npz",
        "beta=6.00_161111_132916.npz",
        # "beta=0.00_161111_143949.npz",
        # "beta=1.00_161111_144002.npz",
        # "beta=2.00_161111_144011.npz",
        # "beta=3.00_161111_144019.npz",
        # "beta=4.00_161111_144025.npz",
        # "beta=5.00_161111_144032.npz",
        # "beta=6.00_161111_144038.npz",
    ]

    data_path = [result_data_path_base + f for f in fn]
    result_N_sub(data_path)
