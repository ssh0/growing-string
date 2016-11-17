#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-10-15


from filled_kagome import FilledKagome
import numpy as np
import matplotlib.pyplot as plt
import time
import save_data


if __name__ == '__main__':
    current_time = time.strftime("%y%m%d_%H%M%S")

    L = 2000
    frames = 1000
    num_of_strings = 30

    betas = [0., 5., 10., 15., 20.]
    Rs = []
    for beta in betas:

        R = []
        for s in range(num_of_strings):
            filled_kagome = FilledKagome(beta=beta, L=L, frames=frames)
            R.append(filled_kagome.R)
        Rs.append(np.average(R))

        save_data.save("results/data/filled_kagome_radius/beta=%2.2f_" % beta,
                       beta=beta, num_of_strings=num_of_strings,
                       L=L, frames=frames, R=R)

    fig, ax = plt.subplots()

    # heatmap
    ax.plot(, path_length, bins=25)

    ax.set_xlabel('Distance')
    ax.set_ylabel('Path length')
    ax.set_title('Path length and distances between two points in the cluster'
                + r'($\beta = %2.2f$)' % beta)
