#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-10-12

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D


if __name__ == '__main__':

    # result_data_path = "./results/data/distances/beta=0.00_161012_171430.npz"
    # result_data_path = "./results/data/distances/beta=5.00_161012_171649.npz"
    # result_data_path = "./results/data/distances/beta=10.00_161012_172119.npz"
    # result_data_path = "./results/data/distances/beta=15.00_161012_172209.npz"
    # result_data_path = "./results/data/distances/beta=20.00_161012_172338.npz"

    # after modifying the method of calcurating the weights
    result_data_path = "./results/data/distances/beta=0.00_161015_153311.npz"

    data = np.load(result_data_path)
    beta = data['beta']
    num_of_strings = data['num_of_strings']
    L = data['L']
    frames = data['frames']
    distance_list = data['distance_list']
    path_length = data['path_length']

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    hist, xedges, yedges = np.histogram2d(distance_list, path_length, bins=100)
    xpos, ypos = np.meshgrid(xedges[:-1] + (xedges[1] - xedges[0]) / 2.,
                             yedges[:-1] + (yedges[1] - yedges[0]) / 2.)
    zpos = hist.T
    ax.plot_wireframe(xpos, ypos, zpos, rstride=1)
    ax.plot(xpos[0], xpos[0], lw=2)

    ax.set_aspect('equal')
    ax.set_xlim(xedges[0], xedges[-1])
    ax.set_ylim(yedges[0], yedges[-1])
    ax.set_xlabel('Distance')
    ax.set_ylabel('Path length')
    ax.set_title('Path length and distances between two points in the cluster'
                + r'($\beta = %2.2f$)' % beta)
    plt.show()
