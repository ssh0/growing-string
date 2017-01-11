#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2017-01-07


from cutting_profile import CuttingProfile
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.cm as cm
import numpy as np
from tqdm import tqdm
import save_data as sd
import argparse


def main(num_of_strings=30, beta=0., frames=1000, L=100, save_result=True,
         plot_result=True):
    params = {
        'beta': beta,
        'L': L,
        'frames': frames,
        'save_result': False,
        'plot_raw_result': False,
        '_plot_dist_to_verify': False,
    }
    relative_positions = {}
    for i in tqdm(range(num_of_strings)):
        runner = CuttingProfile(**params)
        runner.start()
        for j in range(6):
            if runner.relative_positions.has_key(j):
                if not relative_positions.has_key(j):
                    relative_positions[j] = runner.relative_positions[j]
                else:
                    relative_positions[j] = np.vstack(
                        (relative_positions[j], runner.relative_positions[j]))

    if save_result:
        sd.save("results/data/cutting_profile/" +
                "frames=%d_beta=%2.2f_" % (frames, beta),
                beta=beta, L=L, frames=frames,
                weight_const=runner.weight_const,
                num_of_strings=num_of_strings,
                relative_positions=relative_positions
                )

    if plot_result:
        # plot_all_points(relative_positions)
        plot_hist(relative_positions)

def plot_hist(relative_positions, save_image=False):

    cmap = cm.viridis
    for k in range(6):
        if relative_positions.has_key(k):
            fig, ax = plt.subplots()
            x, y = relative_positions[k].T
            ax.hist2d(x, y, bins=50, cmap=cmap)
            ax.set_aspect('equal')
            ax.set_title('vec: {}'.format(k))
            if save_image:
                save_fig(fig, fn='hist_vec={}_'.format(k))
            else:
                plt.show()
            plt.close()

def plot_all_points(relative_positions, save_image=False):
    fig, ax = plt.subplots()
    max_width = 0
    max_height = 0
    for k in range(6):
        if relative_positions.has_key(k):
            x, y = relative_positions[k].T
            _max_width = np.max(np.abs(x))
            _max_height = np.max(np.abs(y))
            if _max_width > max_width:
                max_width = _max_width
            if _max_height > max_height:
                max_height = _max_height
            ax.plot(x, y, marker='.', ls='none', alpha=0.8,
                    label='vec: {}'.format(k),
                    color=cm.rainbow(float(k) / 5))

    ax.set_xlim((-max_width, max_width))
    ax.set_ylim((-max_height, max_height))
    ax.legend(loc='best')
    if save_image:
        save_fig(fig)
    else:
        plt.show()

def plot_3d_wireframe(relative_positions, bins=30, save_image=False):
    for k in range(6):
        if relative_positions.has_key(k):
            fig, ax = plt.subplots()
            ax = fig.add_subplot(111, projection='3d')
            x, y = relative_positions[k].T
            max_width = np.max(np.abs(x))
            max_height = np.max(np.abs(y))

            hist, xedges, yedges = np.histogram2d(x, y, bins=bins)
            X, Y = np.meshgrid(xedges[:-1]+xedges[1:], yedges[:-1]+yedges[1:])
            X, Y = X/2., Y/2.
            ax.plot_wireframe(X, Y, hist)
            ax.set_title('vec: {}'.format(k))
            ax.set_xlabel(r'$x$')
            ax.set_ylabel(r'$y$')
            ax.set_zlabel(r'$f$')
            ax.set_xlim((-max_width, max_width))
            ax.set_ylim((-max_height, max_height))
            if save_image:
                save_fig(fig)
            else:
                plt.show()
            plt.close()

def save_fig(fig, fn=''):
    import time
    current_time = time.strftime("%y%m%d_%H%M%S")
    base_dir = './results/img/cutting_profile/'
    filename = base_dir + fn + current_time + '.png'
    fig.savefig(filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('beta', type=float, nargs=1,
                        help='parameter beta (inverse temparature)')
    args = parser.parse_args()
    beta = args.beta[0]

    print "beta = %2.2f" % beta
    params = {
        'num_of_strings': 1000,
        'beta': beta,
        'frames': 1000,
        'L': 1000 * 2 + 2,
        'save_result': True,
        'plot_result': False,
    }
    main(**params)
