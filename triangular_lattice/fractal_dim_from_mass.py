#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2017-01-21


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
from optimize import Optimize_powerlaw
import time


def load_data(path):
    data = np.load(path)
    beta = data['beta']
    num_of_strings = data['num_of_strings']
    frames = data['frames']
    N_r = data['N_r']
    r = data['r']
    M = data['M']
    return float(beta), int(num_of_strings), int(N_r), int(frames), r, M

def _plot_data_for_validation(paths, raw=False):
    fig, ax = plt.subplots()
    if raw:  # Plot raw data
        for path in paths:
            beta, num_of_strings, N_r, frames, r, M = load_data(path)
            ax.loglog(r, M, '.',
                      label=r'$\beta = %2.2f$, $T = %d$' % (beta, frames))
        ax.set_title('Mass in the circle of radius $r$')
        ax.set_ylabel(r'Mass in the circle of radius')
    else:  # Plot averaged data
        for path in paths:
            beta, num_of_strings, N_r, frames, r, M = load_data(path)
            r, M = averaging_data(r, M, N_r, scale='log')
            ax.loglog(r, M, '.',
                      label=r'$\beta = %2.2f$, $T = %d' % (beta, frames))
        ax.set_title('Averaged mass in the circle of radius $r$')
        ax.set_ylabel(r'Averaged mass in the circle of radius')
    ax.legend(loc='best')
    ax.set_aspect('equal')
    ax.set_xlabel(r'Radius $r$')
    plt.show()

def averaging_data(x, y, x_bin, scale='linear'):
    x_min, x_max = np.min(x), np.max(x)
    if scale == 'linear':
        x_width = (x_max - x_min) / float(x_bin)
        x_edges = [x_min + x_width * i for i in range(x_bin + 1)]
    elif scale == 'log':
        x_width_log = (np.log(x_max) - np.log(x_min)) / float(x_bin)
        x_edges = [np.exp(np.log(x_min) + x_width_log * i) for i in range(x_bin)]
    else:
        raise AttributeError("option `scale` must be 'linear' or 'log'")
    X, Y = [], []
    for left, right in zip(x_edges[:-1], x_edges[1:]):
        index = np.where((x >= left) & (x < right))[0]  # x_max のデータは除かれる?
        if len(index) == 0:
            continue
        _X = np.average(x[index])
        _Y = np.average(y[index])
        X.append(_X)
        Y.append(_Y)
    return np.array(X), np.array(Y)

def get_fractal_dim(path):
    beta, num_of_strings, N_r, frames, r, M = load_data(path)
    fig, ax = plt.subplots()
    r, M = averaging_data(r, M, N_r, scale='log')
    ax.loglog(r, M, '.')
    ax.set_aspect('equal')
    ax.set_title(r'Averaged mass in the circle ' +
                 r'($\beta = {}$, $T = {}$)'.format(beta, frames))
    ax.set_xlabel(r'Radius $r$')
    ax.set_ylabel(r'Averaged mass in the circle of radius')

    def onselect(vmin, vmax):
        global result, selected_index, ln, text
        if globals().has_key('ln') and ln:
            ln.remove()
            text.remove()

        selected_index = np.where((r >= vmin) & (r <= vmax))
        optimizer = Optimize_powerlaw(
            args=(r[selected_index], M[selected_index]),
            parameters=[1., 0.5])
        result = optimizer.fitting()
        D = result['D']
        print "beta = {}, D = {}".format(beta, D)
        optimizer.c = result['c'] + 1.
        X = r[selected_index]
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
            fn = "results/img/mass_in_r/frames=%d_beta=%2.2f" % (frames, beta)
            fn += "_" + time.strftime("%y%m%d_%H%M%S") + ".png"
            plt.savefig(fn)
            print "[saved] " + fn
            plt.close()

    span = SpanSelector(ax, onselect, direction='horizontal')
    fig.canvas.mpl_connect('key_press_event', press)
    plt.show()

def get_paths(fix=None, beta_num=0, frame_num=0):
    """get specific condtion datas
    filter: 'beta' or 'frames'
    """
    # ls -1 ./results/data/mass_in_r/beta=0.00_frames=*.npz | sort -V
    # ls -1 ./results/data/mass_in_r/beta=2.00_frames=*.npz | sort -V
    # ls -1 ./results/data/mass_in_r/beta=4.00_frames=*.npz | sort -V
    # ls -1 ./results/data/mass_in_r/beta=6.00_frames=*.npz | sort -V
    # ls -1 ./results/data/mass_in_r/beta=8.00_frames=*.npz | sort -V
    # ls -1 ./results/data/mass_in_r/beta=10.00_frames=*.npz | sort -V
    result_data_paths = [
        "./results/data/mass_in_r/beta=0.00_frames=200_170122_014239.npz",
        "./results/data/mass_in_r/beta=0.00_frames=400_170122_014239.npz",
        "./results/data/mass_in_r/beta=0.00_frames=600_170122_014240.npz",
        "./results/data/mass_in_r/beta=0.00_frames=800_170122_014240.npz",
        "./results/data/mass_in_r/beta=0.00_frames=1000_170122_014240.npz",
        "./results/data/mass_in_r/beta=0.00_frames=1200_170122_014240.npz",
        "./results/data/mass_in_r/beta=0.00_frames=1400_170122_014240.npz",
        "./results/data/mass_in_r/beta=0.00_frames=1600_170122_014240.npz",
        "./results/data/mass_in_r/beta=0.00_frames=1800_170122_014240.npz",
        "./results/data/mass_in_r/beta=0.00_frames=2000_170122_014240.npz",

        "./results/data/mass_in_r/beta=2.00_frames=200_170122_015933.npz",
        "./results/data/mass_in_r/beta=2.00_frames=400_170122_015933.npz",
        "./results/data/mass_in_r/beta=2.00_frames=600_170122_015933.npz",
        "./results/data/mass_in_r/beta=2.00_frames=800_170122_015933.npz",
        "./results/data/mass_in_r/beta=2.00_frames=1000_170122_015933.npz",
        "./results/data/mass_in_r/beta=2.00_frames=1200_170122_015933.npz",
        "./results/data/mass_in_r/beta=2.00_frames=1400_170122_015933.npz",
        "./results/data/mass_in_r/beta=2.00_frames=1600_170122_015933.npz",
        "./results/data/mass_in_r/beta=2.00_frames=1800_170122_015933.npz",
        "./results/data/mass_in_r/beta=2.00_frames=2000_170122_015933.npz",

        "./results/data/mass_in_r/beta=4.00_frames=200_170122_023211.npz",
        "./results/data/mass_in_r/beta=4.00_frames=400_170122_023211.npz",
        "./results/data/mass_in_r/beta=4.00_frames=600_170122_023211.npz",
        "./results/data/mass_in_r/beta=4.00_frames=800_170122_023211.npz",
        "./results/data/mass_in_r/beta=4.00_frames=1000_170122_023211.npz",
        "./results/data/mass_in_r/beta=4.00_frames=1200_170122_023211.npz",
        "./results/data/mass_in_r/beta=4.00_frames=1400_170122_023211.npz",
        "./results/data/mass_in_r/beta=4.00_frames=1600_170122_023211.npz",
        "./results/data/mass_in_r/beta=4.00_frames=1800_170122_023211.npz",
        "./results/data/mass_in_r/beta=4.00_frames=2000_170122_023211.npz",

        "./results/data/mass_in_r/beta=6.00_frames=200_170122_025607.npz",
        "./results/data/mass_in_r/beta=6.00_frames=400_170122_025607.npz",
        "./results/data/mass_in_r/beta=6.00_frames=600_170122_025607.npz",
        "./results/data/mass_in_r/beta=6.00_frames=800_170122_025607.npz",
        "./results/data/mass_in_r/beta=6.00_frames=1000_170122_025607.npz",
        "./results/data/mass_in_r/beta=6.00_frames=1200_170122_025607.npz",
        "./results/data/mass_in_r/beta=6.00_frames=1400_170122_025607.npz",
        "./results/data/mass_in_r/beta=6.00_frames=1600_170122_025607.npz",
        "./results/data/mass_in_r/beta=6.00_frames=1800_170122_025607.npz",
        "./results/data/mass_in_r/beta=6.00_frames=2000_170122_025607.npz",

        "./results/data/mass_in_r/beta=8.00_frames=200_170122_032301.npz",
        "./results/data/mass_in_r/beta=8.00_frames=400_170122_032301.npz",
        "./results/data/mass_in_r/beta=8.00_frames=600_170122_032301.npz",
        "./results/data/mass_in_r/beta=8.00_frames=800_170122_032301.npz",
        "./results/data/mass_in_r/beta=8.00_frames=1000_170122_032301.npz",
        "./results/data/mass_in_r/beta=8.00_frames=1200_170122_032301.npz",
        "./results/data/mass_in_r/beta=8.00_frames=1400_170122_032301.npz",
        "./results/data/mass_in_r/beta=8.00_frames=1600_170122_032301.npz",
        "./results/data/mass_in_r/beta=8.00_frames=1800_170122_032301.npz",
        "./results/data/mass_in_r/beta=8.00_frames=2000_170122_032301.npz",

        "./results/data/mass_in_r/beta=10.00_frames=200_170122_033138.npz",
        "./results/data/mass_in_r/beta=10.00_frames=400_170122_033138.npz",
        "./results/data/mass_in_r/beta=10.00_frames=600_170122_033138.npz",
        "./results/data/mass_in_r/beta=10.00_frames=800_170122_033138.npz",
        "./results/data/mass_in_r/beta=10.00_frames=1000_170122_033138.npz",
        "./results/data/mass_in_r/beta=10.00_frames=1200_170122_033138.npz",
        "./results/data/mass_in_r/beta=10.00_frames=1400_170122_033138.npz",
        "./results/data/mass_in_r/beta=10.00_frames=1600_170122_033138.npz",
        "./results/data/mass_in_r/beta=10.00_frames=1800_170122_033138.npz",
        "./results/data/mass_in_r/beta=10.00_frames=2000_170122_033138.npz",
    ]

    if fix == 'beta':  # fix beta (all frames)
        result_data_paths = [result_data_paths[beta_num * 10 + i]
                             for i in range(10)]
    elif fix == 'frames':  # fix frames (all beta)
        result_data_paths = [result_data_paths[i * 10 + frame_num]
                             for i in range(6)]
    elif fix is None:
        result_data_paths = [result_data_paths[beta_num * 10 + frame_num]]
    return result_data_paths


if __name__ == '__main__':
    ## frame = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
    ##          0    1    2    3    4     5     6     7     8     9
    ## beta = [0, 2, 4, 6, 8, 10]
    ##         0  1  2  3  4  5

    # result_data_paths = get_paths(beta_num=5, frame_num=9)
    # result_data_paths = get_paths(fix='frames', frame_num=7)
    # result_data_paths = get_paths(fix='beta', beta_num=0)

    # _plot_data_for_validation(result_data_paths, raw=True)
    # _plot_data_for_validation(result_data_paths)

    for path in result_data_paths:
        get_fractal_dim(path)

