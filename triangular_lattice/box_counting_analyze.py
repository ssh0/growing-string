#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-11-17

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.cm as cm
import numpy as np
import itertools
from scipy.optimize import curve_fit
from scipy.stats import gamma

result_data_path_base = "./results/data/box_counting/"
fn = [
# ls ./results/data/box_counting/
    "beta=0.00_161117_140700.npz",
    "beta=1.00_161117_140528.npz",
    "beta=1.00_161117_140704.npz",
    "beta=2.00_161117_140709.npz",
    "beta=3.00_161117_140714.npz",
    "beta=4.00_161117_140720.npz",
    "beta=5.00_161117_140725.npz",
    "beta=6.00_161117_140734.npz",
    "beta=7.00_161117_152439.npz",
    "beta=8.00_161117_152444.npz",
    "beta=9.00_161117_152448.npz",
    "beta=10.00_161117_152454.npz",
]

def plot_Ds():
    fig, ax = plt.subplots()
    fpath = [result_data_path_base + f for f in fn]
    for i, result_data_path in enumerate(fpath):
        data = np.load(result_data_path)
        beta = data['beta']
        L = data['L']
        frames = data['frames']
        Ds = data['Ds']
        T = np.arange(frames)

        ax.semilogx(np.sqrt(T), Ds, '.', label=r'$\beta = %2.2f$' % beta,
                color=cm.viridis(float(i) / len(fpath)))

    ax.legend(loc='best')
    ax.set_title('Fractal dimension')
    ax.set_xlabel(r'$T$')
    ax.set_ylabel(r'$D(T)$')

    plt.show()


if __name__ == '__main__':
    plot_Ds()
