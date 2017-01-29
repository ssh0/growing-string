#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2017-01-27

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
    N_L = data['N_L']
    Ls = data['Ls']
    N = data['N']
    return float(beta), int(num_of_strings), int(N_L), int(frames), Ls, N

def _plot_data_for_validation(paths, raw=False):
    fig, ax = plt.subplots()
    if raw:  # Plot raw data
        for path in paths:
            beta, num_of_strings, N_L, frames, Ls, N = load_data(path)
            ax.loglog(Ls, N, '.',
                      label=r'$\beta = %2.2f$, $T = %d$' % (beta, frames))
        ax.set_title(r'Box count')
        ax.set_ylabel(r'$N(\delta)$')
    else:  # Plot averaged data
        for path in paths:
            beta, num_of_strings, N_r, frames, Ls, N = load_data(path)
            Ls, N = averaging_data(Ls, N, N_r, scale='log')
            ax.loglog(Ls, N, '.',
                      label=r'$\beta = %2.2f$, $T = %d$' % (beta, frames))
        ax.set_title(r'Box count (averaged)')
        ax.set_ylabel(r'$N(\delta)$')
    ax.legend(loc='best')
    ax.set_aspect('equal')
    ax.set_xlabel(r'Cutting size $L$')
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
    beta, num_of_strings, N_r, frames, Ls, N = load_data(path)
    fig, ax = plt.subplots()
    # Ls, N = averaging_data(Ls, N, N_r, scale='log')
    ax.loglog(Ls, N, '.')
    ax.set_aspect('equal')
    ax.set_title(r'Box count' +
                 r'($\beta = {}$, $T = {}$)'.format(beta, frames))
    ax.set_ylabel(r'$N(\delta)$')
    ax.set_xlabel(r'Cutting size $\delta$')

    def onselect(vmin, vmax):
        global result, selected_index, ln, text, D
        if globals().has_key('ln') and ln:
            ln.remove()
            text.remove()

        selected_index = np.where((Ls >= vmin) & (Ls <= vmax))
        optimizer = Optimize_powerlaw(
            args=(Ls[selected_index], N[selected_index]),
            parameters=[1000., -1.5])
        result = optimizer.fitting()
        D = - result['D']
        print "beta = {}, D = {}".format(beta, D)
        optimizer.c = result['c']
        X = Ls[selected_index]
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
            fn = "./results/img/fractal_dim/2017-01-27/raw_frames=%d_beta=%.0f" % (frames, beta)
            fn += "_" + time.strftime("%y%m%d_%H%M%S")
            # fn += ".png"
            fn += ".pdf"
            plt.savefig(fn, bbox_inches='tight')
            print "[saved] " + fn
            plt.close()

    span = SpanSelector(ax, onselect, direction='horizontal')
    fig.canvas.mpl_connect('key_press_event', press)
    fig.tight_layout()
    plt.show()

def get_paths(fix=None, beta_num=0, frame_num=0):
    """get specific condtion datas
    filter: 'beta' or 'frames'
    """
    # ls -1 ./results/data/box_counting/2017-01-27/beta=0.00_frames=*.npz | sort -V
    # ls -1 ./results/data/box_counting/2017-01-27/beta=2.00_frames=*.npz | sort -V
    # ls -1 ./results/data/box_counting/2017-01-27/beta=4.00_frames=*.npz | sort -V
    # ls -1 ./results/data/box_counting/2017-01-27/beta=6.00_frames=*.npz | sort -V
    # ls -1 ./results/data/box_counting/2017-01-27/beta=8.00_frames=*.npz | sort -V
    # ls -1 ./results/data/box_counting/2017-01-27/beta=10.00_frames=*.npz | sort -V

    result_data_paths = [
        "./results/data/box_counting/2017-01-27/beta=0.00_frames=200_170129_033959.npz",
        "./results/data/box_counting/2017-01-27/beta=0.00_frames=400_170129_033959.npz",
        "./results/data/box_counting/2017-01-27/beta=0.00_frames=600_170129_033959.npz",
        "./results/data/box_counting/2017-01-27/beta=0.00_frames=800_170129_033959.npz",
        "./results/data/box_counting/2017-01-27/beta=0.00_frames=1000_170129_033959.npz",
        "./results/data/box_counting/2017-01-27/beta=0.00_frames=1200_170129_033959.npz",
        "./results/data/box_counting/2017-01-27/beta=0.00_frames=1400_170129_033959.npz",
        "./results/data/box_counting/2017-01-27/beta=0.00_frames=1600_170129_033959.npz",
        "./results/data/box_counting/2017-01-27/beta=0.00_frames=1800_170129_033959.npz",
        "./results/data/box_counting/2017-01-27/beta=0.00_frames=2000_170129_033959.npz",
        "./results/data/box_counting/2017-01-27/beta=2.00_frames=200_170129_032930.npz",
        "./results/data/box_counting/2017-01-27/beta=2.00_frames=400_170129_032930.npz",
        "./results/data/box_counting/2017-01-27/beta=2.00_frames=600_170129_032930.npz",
        "./results/data/box_counting/2017-01-27/beta=2.00_frames=800_170129_032930.npz",
        "./results/data/box_counting/2017-01-27/beta=2.00_frames=1000_170129_032930.npz",
        "./results/data/box_counting/2017-01-27/beta=2.00_frames=1200_170129_032930.npz",
        "./results/data/box_counting/2017-01-27/beta=2.00_frames=1400_170129_032930.npz",
        "./results/data/box_counting/2017-01-27/beta=2.00_frames=1600_170129_032930.npz",
        "./results/data/box_counting/2017-01-27/beta=2.00_frames=1800_170129_032930.npz",
        "./results/data/box_counting/2017-01-27/beta=2.00_frames=2000_170129_032930.npz",
        "./results/data/box_counting/2017-01-27/beta=4.00_frames=200_170129_034118.npz",
        "./results/data/box_counting/2017-01-27/beta=4.00_frames=400_170129_034118.npz",
        "./results/data/box_counting/2017-01-27/beta=4.00_frames=600_170129_034118.npz",
        "./results/data/box_counting/2017-01-27/beta=4.00_frames=800_170129_034118.npz",
        "./results/data/box_counting/2017-01-27/beta=4.00_frames=1000_170129_034118.npz",
        "./results/data/box_counting/2017-01-27/beta=4.00_frames=1200_170129_034118.npz",
        "./results/data/box_counting/2017-01-27/beta=4.00_frames=1400_170129_034118.npz",
        "./results/data/box_counting/2017-01-27/beta=4.00_frames=1600_170129_034118.npz",
        "./results/data/box_counting/2017-01-27/beta=4.00_frames=1800_170129_034118.npz",
        "./results/data/box_counting/2017-01-27/beta=4.00_frames=2000_170129_034118.npz",
        "./results/data/box_counting/2017-01-27/beta=6.00_frames=200_170129_041933.npz",
        "./results/data/box_counting/2017-01-27/beta=6.00_frames=400_170129_041933.npz",
        "./results/data/box_counting/2017-01-27/beta=6.00_frames=600_170129_041933.npz",
        "./results/data/box_counting/2017-01-27/beta=6.00_frames=800_170129_041933.npz",
        "./results/data/box_counting/2017-01-27/beta=6.00_frames=1000_170129_041933.npz",
        "./results/data/box_counting/2017-01-27/beta=6.00_frames=1200_170129_041933.npz",
        "./results/data/box_counting/2017-01-27/beta=6.00_frames=1400_170129_041933.npz",
        "./results/data/box_counting/2017-01-27/beta=6.00_frames=1600_170129_041933.npz",
        "./results/data/box_counting/2017-01-27/beta=6.00_frames=1800_170129_041933.npz",
        "./results/data/box_counting/2017-01-27/beta=6.00_frames=2000_170129_041933.npz",
        "./results/data/box_counting/2017-01-27/beta=8.00_frames=200_170129_025417.npz",
        "./results/data/box_counting/2017-01-27/beta=8.00_frames=400_170129_025417.npz",
        "./results/data/box_counting/2017-01-27/beta=8.00_frames=600_170129_025417.npz",
        "./results/data/box_counting/2017-01-27/beta=8.00_frames=800_170129_025417.npz",
        "./results/data/box_counting/2017-01-27/beta=8.00_frames=1000_170129_025417.npz",
        "./results/data/box_counting/2017-01-27/beta=8.00_frames=1200_170129_025417.npz",
        "./results/data/box_counting/2017-01-27/beta=8.00_frames=1400_170129_025417.npz",
        "./results/data/box_counting/2017-01-27/beta=8.00_frames=1600_170129_025417.npz",
        "./results/data/box_counting/2017-01-27/beta=8.00_frames=1800_170129_025417.npz",
        "./results/data/box_counting/2017-01-27/beta=8.00_frames=2000_170129_025417.npz",
        "./results/data/box_counting/2017-01-27/beta=10.00_frames=200_170129_035850.npz",
        "./results/data/box_counting/2017-01-27/beta=10.00_frames=400_170129_035850.npz",
        "./results/data/box_counting/2017-01-27/beta=10.00_frames=600_170129_035850.npz",
        "./results/data/box_counting/2017-01-27/beta=10.00_frames=800_170129_035850.npz",
        "./results/data/box_counting/2017-01-27/beta=10.00_frames=1000_170129_035850.npz",
        "./results/data/box_counting/2017-01-27/beta=10.00_frames=1200_170129_035850.npz",
        "./results/data/box_counting/2017-01-27/beta=10.00_frames=1400_170129_035850.npz",
        "./results/data/box_counting/2017-01-27/beta=10.00_frames=1600_170129_035850.npz",
        "./results/data/box_counting/2017-01-27/beta=10.00_frames=1800_170129_035850.npz",
        "./results/data/box_counting/2017-01-27/beta=10.00_frames=2000_170129_035850.npz",
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
    frames_list = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
    ##             0    1    2    3    4     5     6     7     8     9
    beta_list = [0, 2, 4, 6, 8, 10]
    ##           0  1  2  3  4  5

    # result_data_paths = get_paths(beta_num=0, frame_num=2)
    result_data_paths = get_paths(fix='frames', frame_num=9)

    # result_data_paths = get_paths(fix='beta', beta_num=2)

    # _plot_data_for_validation(result_data_paths, raw=True)
    # _plot_data_for_validation(result_data_paths)

    for path in result_data_paths:
        get_fractal_dim(path)

