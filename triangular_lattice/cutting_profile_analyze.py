#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2017-01-09


from cutting_profile_run import plot_hist, plot_all_points, plot_3d_wireframe
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
import numpy as np
# from tqdm import tqdm
# import save_data as sd
# import argparse


def main(data_path):
    ## Load existing data from data dir
    data = np.load(data_path)
    beta = data['beta']
    L = data['L']
    frames = data['frames']
    weight_const = data['weight_const']
    num_of_strings = data['num_of_strings']
    relative_positions = data['relative_positions'].item()
    num_of_strings = data['num_of_strings']

    # print relative_positions
    ## Plot results
    # plot_all_points(relative_positions)
    # plot_hist(relative_positions)
    plot_3d_wireframe(relative_positions)


if __name__ == '__main__':
    base_dir = './results/data/cutting_profile/'
    fn = [
        'frames=1000_beta=0.00_170108_000219.npz',
        'frames=1000_beta=2.00_170108_004125.npz',
        'frames=1000_beta=4.00_170108_012900.npz',
        'frames=1000_beta=6.00_170108_015700.npz',
        'frames=1000_beta=8.00_170108_020451.npz',
        'frames=1000_beta=10.00_170108_020300.npz',
        # ls -1 ./results/data/cutting_profile/*.npz
    ]

    main(base_dir + fn[0])
