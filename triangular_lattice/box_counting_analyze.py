#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-11-17

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.cm as cm
import numpy as np
import glob
# import itertools
# from scipy.optimize import curve_fit
# from scipy.stats import gamma

# result_data_path_base = "./results/data/box_counting/"
# fn = [
# # ls ./results/data/box_counting/
#     "beta=0.00_161117_140700.npz",
#     "beta=1.00_161117_140704.npz",
#     "beta=2.00_161117_140709.npz",
#     "beta=3.00_161117_140714.npz",
#     "beta=4.00_161117_140720.npz",
#     "beta=5.00_161117_140725.npz",
#     "beta=6.00_161117_140734.npz",
#     "beta=7.00_161117_152439.npz",
#     "beta=8.00_161117_152444.npz",
#     "beta=9.00_161117_152448.npz",
#     "beta=10.00_161117_152454.npz",
# ]
# fpath = [result_data_path_base + f for f in fn]

fpath = sorted(glob.glob('./results/data/box_counting/*.npz'))

def plot_Ds():
    fig, ax = plt.subplots()
    for i, result_data_path in enumerate(fpath):
        data = np.load(result_data_path)
        beta = data['beta']
        L = data['L']
        frames = data['frames']
        Ds = data['Ds']
        T = np.log(0.5 * (np.arange(frames) + 1) ** 2)
        # filtered = np.where(Ds < 1.)
        # Ds[filtered] = 1.

        ax.plot(T, Ds, '.', label=r'$\beta = %2.2f$' % beta,
                color=cm.viridis(float(i) / len(fpath)))

    ax.legend(loc='best')
    ax.set_title('Fractal dimension')
    ax.set_xlabel(r'$T$')
    ax.set_ylabel(r'$D(T)$')

    plt.show()

def plot_Ds_3d():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    D = {}
    for i, result_data_path in enumerate(fpath):
        data = np.load(result_data_path)
        beta = float(data['beta'])
        L = data['L']
        frames = data['frames']
        Ds = data['Ds']
        T = np.log(0.5 * (np.arange(frames) + 1) ** 2)

        if D.has_key(beta):
            D[beta].append(Ds)
        else:
            D[beta] = [Ds]

    # betas = np.array(betas)
    betas = sorted(D.keys())
    D = np.array([np.average(np.array(D[k]), axis=0) for k in betas])
    print D.shape
    X, Y = np.meshgrid(T, betas)
    ax.plot_wireframe(X, Y, D, rstride=1)
    ax.set_title('Fractal dimension')
    ax.set_xlabel(r'$T$')
    ax.set_ylabel(r'$\beta$')
    ax.set_zlabel(r'$D(T)$')

    plt.show()


if __name__ == '__main__':
    # plot_Ds()
    plot_Ds_3d()
