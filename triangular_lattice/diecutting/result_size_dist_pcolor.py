#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-12-06


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.cm as cm
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import gamma
import set_data_path


def result_size_dist_pcolor(path):
    fig, ax = plt.subplots()
    for i, result_data_path in enumerate(path):
        data = np.load(result_data_path)
        beta = data['beta']
        num_of_strings = data['num_of_strings']
        L = data['L']
        frames = data['frames']
        Ls = data['Ls'].astype(np.float)
        size_dist = data['size_dist']
        size_dist = size_dist / np.max(size_dist)
        # size_dist = np.array([_s / np.sum(_s) for _s in size_dist])
        # size_dist = np.array([_s / np.max(_s) for _s in size_dist])
        X, Y = np.meshgrid(Ls, range(len(size_dist[0])))
        ax.pcolor(X, Y, size_dist.T)

    ax.set_title("Histogram of the size of subclusters in the region with $L$")
    ax.set_xlim(0, max(Ls))
    ax.set_ylim(0, size_dist.shape[1])
    # ax.set_ylim(0, size_dist.shape[0] * 2)
    ax.set_xlabel("$L$")
    ax.set_ylabel("Size of a subcluster")
    ax.set_aspect('equal')
    plt.show()


if __name__ == '__main__':
    result_size_dist_pcolor(set_data_path.data_path)  # dataは1つにする
