#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-10-13


import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    fig, ax = plt.subplots()
    for beta in [0., 5., 10., 15., 20.]:
    # for beta in [float(i) for i in range(18)]:

        # result_data_path = "./results/data/correlation/beta=%2.2f_161013_141137.npz" % beta
        # result_data_path = "./results/data/correlation/beta=%2.2f_161013_154842.npz" % beta
        # result_data_path = "./results/data/correlation/beta=%2.2f_161015_155449.npz" % beta
        result_data_path = "./results/data/correlation/beta=%2.2f_161018_220923.npz" % beta


        data = np.load(result_data_path)
        beta = data['beta']
        num_of_strings = data['num_of_strings']
        L = data['L']
        frames = data['frames']
        Lp = data['Lp']
        Cs = data['Cs']

        # ax.plot(Lp, Cs, '.', label=r'$\beta = %2.2f$' % beta)

        # グラフから，偶数番目と奇数番目で値が局在することが分かる。
        # 実際に偶数番目と奇数番目を分けて表示する。
        Lp_even, Lp_odd = Lp[::2], Lp[1::2]
        Cs_even, Cs_odd = Cs[::2], Cs[1::2]
        ax.plot(Lp_even, Cs_even, '.', label=r'(even) $\beta = %2.2f$' % beta)
        ax.plot(Lp_odd, Cs_odd, '.', label=r'(odd) $\beta = %2.2f$' % beta)

    ax.set_xlabel('Path length')
    ax.set_ylabel('Correlation of the vectors')
    ax.set_title('Correlation of the vectors')
    ax.set_ylim(0., ax.get_ylim()[1])
    ax.legend(loc='best')
    plt.show()
