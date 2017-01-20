#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-10-03


import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
from tqdm import tqdm
from growing_string import Main
from radius import calc_radius_of_rotation
import save_data


frames = 2000
L = (frames + 1) * 2

def calc_R(beta, save_data_for_one_cluster=False):
    params = {
        'Lx': L,
        'Ly': L,
        'frames': frames,
        'beta': beta,
        'size': [3,] * 1,
        'plot': False,
        'save_image': False,
        'strings': [{'id': 1, 'x': L/4, 'y': L/2, 'vec': [0, 4]}],
        'pre_function': calc_radius_of_rotation
    }

    main = Main(**params)
    radius_of_rotation = main.pre_func_res

    ## save data (for one cluster)
    if save_data_for_one_cluster:
        base = "./results/data/radius/frames=%d_beta=%2.2f_" % (frames, beta)
        save_data.save(base,
                       frames=frames,
                       beta=beta,
                       L=L,
                       radius_of_rotation=radius_of_rotation)

    return radius_of_rotation

def calc_ave_R(num_of_strings=100):
    R_ave = np.zeros(frames)
    for s in tqdm(range(num_of_strings)):
        R_ave += calc_R(beta)
    R_ave = R_ave / float(num_of_strings)
    base = "./results/data/radius/"
    base += "frames=%d_beta=%2.2f_sample=%d_" % (frames, beta, num_of_strings)
    save_data.save(base,
                   frames=frames,
                   beta=beta,
                   L=L,
                   radius_of_rotation=R_ave)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('beta', type=float, nargs=1, help='parameter beta')
    args = parser.parse_args()
    beta = args.beta[0]

    ## get data for one cluster
    # calc_R(beta, save_data_for_one_cluster=True)

    ## Get averaged values
    calc_ave_R(num_of_strings=200)
