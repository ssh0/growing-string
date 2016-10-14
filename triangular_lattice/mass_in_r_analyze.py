#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-10-12

import numpy as np
import matplotlib.pyplot as plt
from optimize import Optimize_powerlaw


if __name__ == '__main__':

    result_data_path = "./results/data/mass_in_r/beta=0.00_161012_214821.npz"  # 1.978068
    # result_data_path = "./results/data/mass_in_r/beta=5.00_161012_214909.npz"  # 1.870158
    # result_data_path = "./results/data/mass_in_r/beta=10.00_161012_215014.npz"  # 1.755911
    # result_data_path = "./results/data/mass_in_r/beta=15.00_161012_215259.npz"  # 1.715894
    # result_data_path = "./results/data/mass_in_r/beta=20.00_161012_215411.npz"  # 1.275145

    data = np.load(result_data_path)
    beta = data['beta']
    num_of_strings = data['num_of_strings']
    L = data['L']
    frames = data['frames']
    r = data['r']
    M = data['M']

    # fig, ax = plt.subplots()

    fig, ax = plt.subplots()
    ax.loglog(r, M)
    ax.set_xlabel('Radius $r$ from the center of gravity')
    ax.set_ylabel('Mass in a circle with radius $r$')
    ax.set_title('$r$ vs. $M(r)$')

    index_stop = len(r) - 5
    optimizer = Optimize_powerlaw(args=(r[:index_stop],
                                        M[:index_stop]),
                                parameters=[0.1, 2.])
    result = optimizer.fitting()
    print "D = %f" % result['D']
    ax.loglog(r[:index_stop], optimizer.fitted(r[:index_stop]), lw=2,
                label='D = %f' % result['D'])
    ax.legend(loc='best')

    # result_image_path = "results/img/mass_in_r/beta=%2.2f" % beta
    # result_image_path += "_" + time.strftime("%y%m%d_%H%M%S")
    # result_image_path += ".png"
    # plt.savefig(result_image_path)
    # plt.close()
    # print "[saved] " + result_image_path

    plt.show()

