#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-10-13


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


if __name__ == '__main__':
    fig, ax = plt.subplots()

    result_data_paths = [
        ## frame=1000, before modified Lp, strings=100, [0, 4]
        # "./results/data/correlation/beta=0.00_170125_012601.npz",
        # "./results/data/correlation/beta=2.00_170125_013838.npz",
        # "./results/data/correlation/beta=4.00_170125_015505.npz",
        # "./results/data/correlation/beta=6.00_170125_021432.npz",
        # "./results/data/correlation/beta=8.00_170125_023702.npz",
        # "./results/data/correlation/beta=10.00_170125_025938.npz",

        ## frame=1000, after modified Lp, strings=100, [0, 0]
        "./results/data/correlation/beta=0.00_170125_171427.npz",
        "./results/data/correlation/beta=2.00_170125_174836.npz",
        "./results/data/correlation/beta=4.00_170125_182723.npz",
        "./results/data/correlation/beta=6.00_170125_191135.npz",
        "./results/data/correlation/beta=8.00_170125_200849.npz",
        "./results/data/correlation/beta=10.00_170125_210225.npz",
        "./results/data/correlation/beta=12.00_170125_193941.npz",
        "./results/data/correlation/beta=14.00_170125_203620.npz",
        "./results/data/correlation/beta=16.00_170125_212026.npz",
        "./results/data/correlation/beta=18.00_170125_225012.npz",
        "./results/data/correlation/beta=20.00_170125_233341.npz",
    ]

    for result_data_path in result_data_paths:

    # for beta in [0., 5., 10., 15., 20.]:
    # for beta in [float(i) for i in range(18)]:

        # result_data_path = "./results/data/correlation/beta=%2.2f_161013_141137.npz" % beta
        # result_data_path = "./results/data/correlation/beta=%2.2f_161013_154842.npz" % beta
        # result_data_path = "./results/data/correlation/beta=%2.2f_161015_155449.npz" % beta
        # result_data_path = "./results/data/correlation/beta=%2.2f_161018_220923.npz" % beta
        # result_data_path = "./results/data/correlation/beta=%2.2f_161122_152015.npz" % beta

        data = np.load(result_data_path)
        beta = data['beta']
        num_of_strings = data['num_of_strings']
        L = data['L']
        frames = data['frames']
        Lp = data['Lp']
        Cs = data['Cs']

        ax.plot(Lp, Cs, '.',
                color=cm.viridis(float(beta) / (2 * len(result_data_paths))),
                label=r'$\beta = %2.2f$' % beta)

        # グラフから，偶数番目と奇数番目で値が局在することが分かる。
        # 実際に偶数番目と奇数番目を分けて表示する。
        # Lp_even, Lp_odd = Lp[::2], Lp[1::2]
        # Cs_even, Cs_odd = Cs[::2], Cs[1::2]
        # ax.plot(Lp_even, Cs_even, '.', label=r'(even) $\beta = %2.2f$' % beta)
        # ax.plot(Lp_odd, Cs_odd, '.', label=r'(odd) $\beta = %2.2f$' % beta)

    fig.subplots_adjust(right=0.8)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    ax.set_xlabel('Path length')
    ax.set_ylabel('Correlation of the vectors')
    ax.set_title('Correlation of the vectors')
    ax.set_ylim(ax.get_ylim()[0], 1.001)
    plt.show()
