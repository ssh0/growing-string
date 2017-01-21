#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-08-19


from growing_string import Main
from optimize import Optimize_powerlaw
from surface import get_surface_points, set_labels, get_labeled_position
from matplotlib.widgets import SpanSelector
import matplotlib.pyplot as plt
import numpy as np
import time


class Roughness(Main):
    def __init__(self, L=60, frames=1000, beta=0.):
        Main.__init__(self, Lx=L, Ly=L, size=[3,] * 1, plot=False,
                      frames=frames,
                      beta=beta,
                      strings=[{'id': 1, 'x': L/4, 'y': L/2, 'vec': [0, 4]}]
                      )

def eval_fluctuation_on_surface(self, pos, test=False):
    position = get_surface_points(self, pos)
    label_lattice = set_labels(self, position)
    label_list = label_lattice[position]
    index = get_labeled_position(self, pos, test)

    X = np.average(self.lattice_X[index])
    Y = np.average(self.lattice_Y[index])
    x = self.lattice_X[index] - X
    y = self.lattice_Y[index] - Y
    r = np.sqrt(x ** 2 + y ** 2)
    R = np.sqrt(np.average(x ** 2 + y ** 2))
    theta = np.arctan(y / x)
    theta[x < 0] = theta[x < 0] + np.pi
    theta = theta % (2 * np.pi)
    label_list = label_lattice[index]
    return np.array([theta, r]), R, label_list

def plot_to_verify(beta, frames, theta, r, R_t, label_list, save_image=False):
    fig = plt.figure()
    ax_left = fig.add_subplot(121)
    ax_right = fig.add_subplot(122)

    for theta_, r_, label in zip(theta, r, label_list):
        x1, y1 = R_t * np.cos(theta_), R_t * np.sin(theta_)
        x2, y2 = r_ * np.cos(theta_), r_ * np.sin(theta_)
        ax_left.plot([x1, x2], [y1, y2], color='k', alpha=0.5)
        # ax_left.text(x2, y2, label)

    th = np.linspace(0., 2 * np.pi, num=100)
    ax_left.plot(R_t * np.cos(th), R_t * np.sin(th), 'k', alpha=0.5)
    ax_right.plot([0., 2 * np.pi * R_t], [0., 0.], 'k', alpha=0.5)

    for label in sorted(list(set(label_list))):
        index = np.where(label_list == label)[0]
        ax_left.plot(r[index] * np.cos(theta[index]),
                     r[index] * np.sin(theta[index]),
                     '.')
        ax_right.plot(R_t * theta[index], r[index] - R_t)

    ax_left.set_aspect('equal')
    ax_left.set_title(r"Real space ($\beta = {}$, $T = {}$)".format(beta, frames))
    ax_right.set_title('Fluctuation of surface')
    ax_right.set_xlabel(r'$ R \theta$')
    ax_right.set_ylabel(r'$r_{i} - R$')
    if save_image:
        fn = "results/img/roughness/raw_frames=%d_beta=%2.2f" % (frames, beta)
        fn += "_" + time.strftime("%y%m%d_%H%M%S") + ".png"
        plt.savefig(fn)
        print "[saved] " + fn

def eval_std_various_width(theta, r, R_t):
    L = theta * R_t
    L_max = 2. * np.pi * R_t

    res_width = []
    res_std = []
    width_sample = 50
    samples_N = 100

    log_width_min = np.log2(L_max / len(L)) + 1.
    log_width_max = np.log2(L_max)
    for width in np.logspace(log_width_min, log_width_max,
                             base=2., num=width_sample):
        stds = []
        for samples_start in np.linspace(0., L_max - width, num=samples_N):
            try:
                index_start = np.min(np.where(L > samples_start)[0])
                index_end = np.max(np.where(L < samples_start + width)[0])
            except ValueError:
                continue

            # if there are no points in
            if index_start > index_end:
                continue

            h = np.average(np.array(r[index_start:index_end + 1]))
            w = np.sqrt(np.sum((r[index_start:index_end + 1] - h) ** 2) / float(width))
            stds.append(w)

        if len(stds) == 0:
            continue
        else:
            res_width.append(width)
            res_std.append(np.mean(stds))

    return res_width, res_std

def plot_result(beta, frames, x, y, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.loglog(x, y, 'o-')
    # ax.semilogx(x, y, 'o-')
    # ax.semilogy(x, y, 'o-')
    ax.set_title(r"Roughness (averaged) at some width"
                 + r"($\beta = {}$, $T = {}$)".format(beta, frames))
    ax.set_xlabel(r'width')
    ax.set_ylabel(r'$\sigma$')
    ax.set_aspect('equal')
    return ax

def fitting_manual(fig, ax, x, y):
    x = np.array(x)
    y = np.array(y)

    def onselect(vmin, vmax):
        global result, selected_index, ln, text
        if globals().has_key('ln') and ln:
            ln.remove()
            text.remove()

        selected_index = np.where((x >= vmin) & (x <= vmax))
        optimizer = Optimize_powerlaw(
            args=(x[selected_index], y[selected_index]),
            parameters=[1., 0.5])
        result = optimizer.fitting()
        D = result['D']
        print result['D']
        # print "beta = {}, D = {}".format(beta, D)
        optimizer.c = result['c'] + 0.15
        X = x[selected_index]
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
            fn = "results/img/roughness/frames=%d_beta=%2.2f" % (frames, beta)
            fn += "_" + time.strftime("%y%m%d_%H%M%S") + ".png"
            plt.savefig(fn)
            print "[saved] " + fn
            plt.close()

    span = SpanSelector(ax, onselect, direction='horizontal')
    fig.canvas.mpl_connect('key_press_event', press)
    plt.show()

def start(beta, frames, plot_to_verify_bool=True, save_image_bool=False):
    L = (frames + 1) * 2
    main = Roughness(L=L, frames=frames, beta=beta)

    ## 隣接格子点に同じラベルを振る
    ## すべてのラベル
    # (theta, r), R_t, label_list = eval_fluctuation_on_surface(
    #     main, main.strings[0].pos, test=True)
    ## 最大クラスターのみ表示
    (theta, r), R_t, label_list = eval_fluctuation_on_surface(
        main, main.strings[0].pos, test=False)

    i_srted = np.argsort(theta)
    theta, r, label_list = theta[i_srted], r[i_srted], label_list[i_srted]

    ## plot to verify
    if plot_to_verify_bool:
        plot_to_verify(beta, frames, theta, r, R_t, label_list, save_image_bool)
        plt.show()

    ## Plot the relation between width and std.
    fig, ax = plt.subplots()
    res_width, res_std = eval_std_various_width(theta, r, R_t)
    ax = plot_result(beta, frames, res_width, res_std, ax)
    fitting_manual(fig, ax, res_width, res_std)

if __name__ == '__main__':
    frames = 1000
    # start(2., frames, plot_to_verify_bool=True, save_image_bool=False)

    for beta in [0., 2., 4., 6., 8., 10.]:
        start(beta, frames, plot_to_verify_bool=True, save_image_bool=False)

