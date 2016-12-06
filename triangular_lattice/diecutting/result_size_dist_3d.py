#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-12-06

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.cm as cm
import numpy as np
import set_data_path


def result_size_dist_3d(path):
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    for i, result_data_path in enumerate(path):
        data = np.load(result_data_path)
        beta = data['beta']
        num_of_strings = data['num_of_strings']
        L = data['L']
        frames = data['frames']
        Ls = data['Ls'].astype(np.float)
        size_dist = data['size_dist']
        X, Y = np.meshgrid(Ls, range(len(size_dist[0])))
        ax.plot_wireframe(X, Y, size_dist.T, rstride=1, alpha=0.3,
                        color=cm.gnuplot(float(i) / len(path)))

    ax.set_title("Histogram of the size of subclusters in the region with $L$")
    ax.set_xlabel("$L$")
    ax.set_ylabel("Size of a subcluster")
    ax.set_zlabel("Freq")
    plt.show()


if __name__ == '__main__':
    result_size_dist_3d(set_data_path.data_path)

