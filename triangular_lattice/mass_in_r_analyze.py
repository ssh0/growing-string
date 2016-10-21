#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-10-12

import numpy as np
import matplotlib.pyplot as plt
from optimize import Optimize_powerlaw
import time


if __name__ == '__main__':
    current_time = time.strftime("%y%m%d_%H%M%S")

    result_data_paths = [
        # "./results/data/mass_in_r/beta=0.00_161012_214821.npz",  # 1.978068
        # "./results/data/mass_in_r/beta=5.00_161012_214909.npz",  # 1.870158
        # "./results/data/mass_in_r/beta=10.00_161012_215014.npz",  # 1.755911
        # "./results/data/mass_in_r/beta=15.00_161012_215259.npz",  # 1.715894
        # "./results/data/mass_in_r/beta=20.00_161012_215411.npz",  # 1.275145
        "./results/data/mass_in_r/beta=0.00_161018_155958.npz",
        "./results/data/mass_in_r/beta=1.00_161018_160933.npz",
        "./results/data/mass_in_r/beta=2.00_161018_162537.npz",
        "./results/data/mass_in_r/beta=3.00_161018_163857.npz",
        "./results/data/mass_in_r/beta=4.00_161018_170119.npz",
        "./results/data/mass_in_r/beta=5.00_161018_171624.npz",
        "./results/data/mass_in_r/beta=6.00_161018_172839.npz",
        "./results/data/mass_in_r/beta=7.00_161018_174537.npz",
        "./results/data/mass_in_r/beta=8.00_161018_175748.npz",
        "./results/data/mass_in_r/beta=9.00_161018_181035.npz",
        "./results/data/mass_in_r/beta=10.00_161018_182357.npz",
    ]

    fig, ax = plt.subplots()

    Ds = []

    for result_data_path in result_data_paths:
        data = np.load(result_data_path)
        beta = data['beta']
        num_of_strings = data['num_of_strings']
        L = data['L']
        frames = data['frames']
        r = data['r']
        M = data['M']

        # ax.loglog(r, M, label=r'$\beta = %2.2f$' % beta)

        index_stop = len(r) - 20
        optimizer = Optimize_powerlaw(args=(r[:index_stop],
                                            M[:index_stop]),
                                    parameters=[0.1, 2.])
        result = optimizer.fitting()
        print "D = %f" % result['D']
        # ax.loglog(r[:index_stop], optimizer.fitted(r[:index_stop]), lw=2,
        #             label='D = %f' % result['D'])
        Ds.append(result['D'])

    ax.plot(range(11), Ds)
    ax.set_xlabel('Radius $r$ from the center of gravity')
    ax.set_ylabel('Mass in a circle with radius $r$')
    ax.set_title('$r$ vs. $M(r)$')
    # ax.legend(loc='best')

    result_image_path = "results/img/mass_in_r/raw"
    result_image_path += "_" + current_time
    result_image_path += ".png"
    plt.savefig(result_image_path)
    plt.close()
    print "[saved] " + result_image_path

    # plt.show()

