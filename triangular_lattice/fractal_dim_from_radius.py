#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2017-01-20

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
from optimize import Optimize_powerlaw
import time


def load_data(path):
    data = np.load(path)
    beta = data['beta']
    frames = data['frames']
    L = data['L']
    R = data['radius_of_rotation']
    return float(beta), int(frames), int(L), R

def _plot_data_for_validation(paths):
    fig, ax = plt.subplots()
    for path in paths:
        beta, frames, L, R = load_data(path)
        steps = np.arange(frames)
        ax.loglog(steps, R, '.', label=r'$\beta = %2.2f$' % beta)
    ax.legend(loc='best')
    ax.set_title('Radius of rotation')
    ax.set_xlabel(r'Steps $T$')
    ax.set_ylabel(r'Radius of rotation $R$')
    plt.show()

def get_fractal_dim(path):
    beta, frames, L, R = load_data(path)
    steps = np.arange(frames)
    fig, ax = plt.subplots()
    ax.loglog(steps, R, ls='', marker='.')
    ax.set_aspect('equal')
    ax.set_title(r'Radius of rotation ($\beta = {}$, $T = {}$)'.format(beta, frames))
    ax.set_xlabel(r'Steps $T$')
    ax.set_ylabel(r'Radius of rotation $R$')

    def onselect(vmin, vmax):
        global result, selected_index, ln, text
        if globals().has_key('ln') and ln:
            ln.remove()
            text.remove()

        selected_index = np.where((steps >= vmin) & (steps <= vmax))
        optimizer = Optimize_powerlaw(
            args=(steps[selected_index], R[selected_index]),
            parameters=[1., 0.5])
        result = optimizer.fitting()
        D = 1. / result['D']
        print "beta = {}, D = {}".format(beta, D)
        optimizer.c = result['c'] + 0.15
        X = steps[selected_index]
        Y = optimizer.fitted(X)
        ln, = ax.loglog(X, Y, ls='-', marker='', color='k')
        text = ax.text((X[0] + X[-1]) / 2., (Y[0] + Y[-1]) / 2.,
                       r'$D = %2.2f$' % D,
                       ha='center', va='bottom',
                       rotation=np.arctan(result['D']) * (180 / np.pi))

    def press(event):
        global ln
        if event.key == 'a':
            ln = False

        if event.key == 'x':
            # save image
            fn = "results/img/radius/frames=%d_beta=%2.2f" % (frames, beta)
            fn += "_" + time.strftime("%y%m%d_%H%M%S") + ".png"
            plt.savefig(fn)
            print "[saved] " + fn
            plt.close()

    span = SpanSelector(ax, onselect, direction='horizontal')
    fig.canvas.mpl_connect('key_press_event', press)
    plt.show()


if __name__ == '__main__':

    # ls -1 ./results/data/radius/frames=*.npz
    result_data_paths = [
        # "./results/data/radius/frames=2000_beta=0.00_170120_181242.npz",
        # "./results/data/radius/frames=2000_beta=2.00_170120_181259.npz",
        # "./results/data/radius/frames=2000_beta=4.00_170120_181320.npz",
        # "./results/data/radius/frames=2000_beta=6.00_170120_181348.npz",
        # "./results/data/radius/frames=2000_beta=8.00_170120_181404.npz",
        # "./results/data/radius/frames=2000_beta=10.00_170120_181405.npz",

        # "./results/data/radius/frames=2000_beta=0.00_sample=100_170120_191732.npz",
        # "./results/data/radius/frames=2000_beta=2.00_sample=100_170120_193745.npz",
        # "./results/data/radius/frames=2000_beta=4.00_sample=100_170120_201105.npz",
        # "./results/data/radius/frames=2000_beta=6.00_sample=100_170120_203628.npz",
        # "./results/data/radius/frames=2000_beta=8.00_sample=100_170120_210938.npz",
        # "./results/data/radius/frames=2000_beta=10.00_sample=100_170120_212150.npz",

        "./results/data/radius/frames=2000_beta=0.00_sample=200_170120_225734.npz",
        "./results/data/radius/frames=2000_beta=2.00_sample=200_170120_233421.npz",
        "./results/data/radius/frames=2000_beta=4.00_sample=200_170121_003705.npz",
        "./results/data/radius/frames=2000_beta=6.00_sample=200_170121_012543.npz",
        "./results/data/radius/frames=2000_beta=8.00_sample=200_170121_021643.npz",
        "./results/data/radius/frames=2000_beta=10.00_sample=200_170121_024946.npz",
    ]

    # _plot_data_for_validation(result_data_paths)

    get_fractal_dim(result_data_paths[1])

    # for path in result_data_paths:
    #     get_fractal_dim(path)

