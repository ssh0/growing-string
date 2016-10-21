#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-10-11

from growing_string import Main
from mass_in_r import count_point_in_r
from optimize import Optimize_powerlaw
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time


def get_mass_in_r_for_one_string(L, frames, beta, r=None):
    params = {
        'Lx': L,
        'Ly': L,
        'frames': frames,
        'beta': beta,
        'weight_const': 0.,
        'strings': [{'id': 1, 'x': L / 4, 'y': L / 2, 'vec': [0, 4, 2]}, ],
        # 'boundary': {'h': 'reflective', 'v': 'reflective'},
        'boundary': {'h': 'periodic', 'v': 'periodic'},
        'plot': False,
        'plot_surface': False,
        'interval': 1,
    }

    main = Main(**params)
    N_r = 100
    r, M = count_point_in_r(main, main.strings[0], N_r, r)
    return r, M


def mass_in_r_for_one_beta(beta, num_of_strings, L, frames, plot=True,
                           optimize=False, save_image=False, save_data=False):
    print "beta = %2.2f" % beta
    r = None
    rs = []
    Ms = []
    for s in tqdm(range(num_of_strings)):
        r, M = get_mass_in_r_for_one_string(L, frames, beta, r)
        rs.append(r)
        Ms.append(M)

    r = np.average(np.array(rs), axis=0)
    M = np.average(np.array(Ms), axis=0)

    if save_data:
        result_data_path = "results/data/mass_in_r/beta=%2.2f" % beta
        result_data_path += "_" + time.strftime("%y%m%d_%H%M%S")
        result_data_path += ".npz"
        np.savez(result_data_path,
                 num_of_strings=num_of_strings,
                 beta=beta,
                 L=L,
                 frames=frames,
                 r=r,
                 M=M)

    if plot or save_image:
        fig, ax = plt.subplots()
        ax.loglog(r, M)
        ax.set_xlabel('Radius $r$ from the center of gravity')
        ax.set_ylabel('Mass in a circle with radius $r$')
        ax.set_title('$r$ vs. $M(r)$')

        if optimize:
            index_stop = len(r) - 5
            optimizer = Optimize_powerlaw(args=(r[:index_stop],
                                                M[:index_stop]),
                                        parameters=[0., 2.])
            result = optimizer.fitting()
            print "D = %f" % result['D']
            ax.loglog(r[:index_stop], optimizer.fitted(r[:index_stop]), lw=2,
                        label='D = %f' % result['D'])
            ax.legend(loc='best')

        if save_image:
            result_image_path = "results/img/mass_in_r/beta=%2.2f" % beta
            result_image_path += "_" + time.strftime("%y%m%d_%H%M%S")
            result_image_path += ".png"
            plt.savefig(result_image_path)
            plt.close()
            print "[saved] " + result_image_path
        else:
            plt.show()


if __name__ == '__main__':
    # beta = 20.
    # params = {
    #     'num_of_strings': 30,
    #     'L': 2000,
    #     'frames': 1000,
    #     'plot': False,
    #     'optimize': False,
    #     'save_image': False,
    #     'save_data': True,
    # }

    # mass_in_r_for_one_beta(beta, **params)

    for beta in np.arange(11):
        params = {
            'num_of_strings': 30,
            'L': 2000,
            'frames': 1000,
            'plot': False,
            'optimize': False,
            'save_image': False,
            'save_data': True,
        }

        mass_in_r_for_one_beta(beta, **params)

