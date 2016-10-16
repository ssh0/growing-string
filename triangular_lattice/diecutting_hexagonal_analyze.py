#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-10-16

import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':

    result_data_path_base = "./results/data/diecutting/"
    fn = [
        "beta=0.00_161016_190044.npz",
        # "beta=0.10_161016_190056.npz",
        # "beta=0.20_161016_190105.npz",
        # "beta=0.30_161016_190111.npz",
        # "beta=0.40_161016_190125.npz",
        # "beta=0.50_161016_190158.npz",
        # "beta=0.60_161016_190207.npz",
        # "beta=0.70_161016_190217.npz",
        # "beta=0.80_161016_190229.npz",
        # "beta=0.90_161016_190236.npz",
        "beta=1.00_161016_190242.npz",
        "beta=2.00_161016_195423.npz",
        "beta=3.00_161016_195439.npz",
        "beta=4.00_161016_195447.npz",
        "beta=5.00_161016_195452.npz",
        "beta=6.00_161016_195526.npz",
    ]

    fig, ax = plt.subplots()

    for result_data_path in [result_data_path_base + f for f in fn]:
        data = np.load(result_data_path)
        beta = data['beta']
        num_of_strings = data['num_of_strings']
        L = data['L']
        frames = data['frames']
        Ls = data['Ls']
        N_sub = data['N_sub']

        # optimizer = Optimize_linear(
        #     args=(
        #         Ls[:-1],
        #         N_sub[:-1]
        #     ),
        #     parameters=[1.5, 0.]
        # )
        # result = optimizer.fitting()

        ax.plot(Ls, N_sub, '.-', label=r'$\beta = %2.2f$' % beta)
        # ax.plot(
        #     Ls[:-1],
        #     optimizer.fitted(Ls[:-1]),
        #     '-',
        #     label='a = %f' % result['a']
        # )
    ax.legend(loc='best')
    ax.set_ylim((0., 25.))
    ax.set_title('Strings in hexagonal region' +
                 ' (sample: {})'.format(num_of_strings))
    ax.set_xlabel(r'Cutting size $L$')
    ax.set_ylabel('Average number of sub-clusters in a hexagonal region')
    plt.show()


