#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-12-01

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.widgets import SpanSelector
from optimize import Optimize_powerlaw


def calc_fractal_dim_for_each_beta(ax, i, filename):
    data = np.load(filename)
    beta = data['beta']
    num_of_strings = data['num_of_strings']
    L = data['L']
    frames = data['frames']
    distance_list = data['distance_list']
    path_length = data['path_length']

    num_of_pairs = 100

    x = np.array(path_length)
    x = x.reshape((num_of_strings, int(x.shape[0] / num_of_strings / num_of_pairs), num_of_pairs))

    y = np.array(distance_list)
    y = y.reshape((num_of_strings, int(y.shape[0] / num_of_strings / num_of_pairs), num_of_pairs))
    # tortuosity = length_of_the_curve / distance_between_the_ends_of_it
    y_ave = np.average(y, axis=2)

    X = np.average(y_ave, axis=0)
    Y = x[0, :, 0]
    ax.loglog(X, Y, ls='', marker='.', label=r'$\beta = %2.2f$' % beta,
              alpha=0.5)
    result = {'D': None}
    # optimizer = Optimize_powerlaw(args=(X[:], Y[:]), parameters=[0.1, 2.])
    # result = optimizer.fitting()
    # print "beta = {}, D = {}".format(beta, result['D'])
    # ax.loglog(X, optimizer.fitted(X), ls='-', marker='', color='k',
    #           label=r'$D = %2.2f$' % result['D'])
    return beta, result['D']

def calc_fractal_dim_for_each_beta_manual(ax, i, filename, save_image=False):
    data = np.load(filename)
    beta = data['beta']
    num_of_strings = data['num_of_strings']
    L = data['L']
    frames = data['frames']
    distance_list = data['distance_list']
    path_length = data['path_length']

    num_of_pairs = 100

    ## path_length に対して distance_list の平均を取る方法
    x = np.array(path_length)
    x = x.reshape((num_of_strings, int(x.shape[0] / num_of_strings / num_of_pairs), num_of_pairs))

    y = np.array(distance_list)
    y = y.reshape((num_of_strings, int(y.shape[0] / num_of_strings / num_of_pairs), num_of_pairs))
    # tortuosity = length_of_the_curve / distance_between_the_ends_of_it
    y_ave = np.average(y, axis=2)

    X = np.average(y_ave, axis=0)
    Y = x[0, :, 0]

    ## distance に対して path_length の平均を取る方法(✘)
    # hist, xedges, yedges = np.histogram2d(distance_list, path_length, bins=(100, 20))
    # xpos, ypos = np.meshgrid(xedges[:-1] + (xedges[1] - xedges[0]) / 2.,
    #                          yedges[:-1] + (yedges[1] - yedges[0]) / 2.)
    # summention = np.sum(hist, axis=1)
    # not_zero = np.where(summention > 0)[0]
    # z_ave = np.dot(hist, ypos.T[0])[not_zero] / summention[not_zero]

    # X = xpos[0][not_zero]
    # Y = z_ave

    ## X: distance, Y: path_length
    ax.loglog(X, Y, ls='', marker='.', label=r'$\beta = %2.2f$' % beta,
              alpha=0.5)
    ax.legend(loc='best')
    ax.set_title('frames = {}, beta = {}'.format(frames, beta))
    def onselect(vmin, vmax):
        global result, selected_index
        ax.cla()
        selected_index = np.where((X >= vmin) & (X <= vmax))
        optimizer = Optimize_powerlaw(
            args=(X[selected_index], Y[selected_index]),
            parameters=[0.1, 2.])
        result = optimizer.fitting()
        print "beta = {}, D = {}".format(beta, result['D'])
        ax.loglog(X, Y, ls='', marker='.', label=r'$\beta = %2.2f$' % beta,
                alpha=0.5)
        ax.loglog(X[selected_index], optimizer.fitted(X[selected_index]),
                  ls='-', marker='', color='k',
                  label=r'$D = %2.2f$' % result['D'])
        ax.legend(loc='best')
        ax.set_title('frames = {}, beta = {}'.format(frames, beta))
    span = SpanSelector(ax, onselect, direction='horizontal')
    plt.show()

    ax.set_xlabel(r'$\delta$')
    ax.set_ylabel(r'Averaged path length $\bar{L}$')
    ax.set_title(r'$\delta$ -- $L$ (frames=%d)' % frames)
    ax.legend(loc='best')

    if save_image:
        result_image_path = "results/img/fractal_dim/d_L_frames=%d_beta=%2.2f" % (frames, beta)
        result_image_path += "_" + time.strftime("%y%m%d_%H%M%S")
        result_image_path += ".png"
        plt.savefig(result_image_path)
        plt.close()
        print "[saved] " + result_image_path

    plt.close()

    return beta, result['D']

def fractal_dims(result_data_path):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    betas, Ds = [], []

    # for i in [0, 5, 10, 15, 20]:
    # for i in [0]:
    for i in [0, 2, 4, 8, 10]:
    # for i in range(11):
        # fn = result_data_path['%d-2' % i]
        fn = result_data_path['%d-3' % i]
        # fn = result_data_path['%d-4' % i]
        beta, D = calc_fractal_dim_for_each_beta(ax1, i, fn)
        betas.append(beta)
        Ds.append(D)

    ax1.set_ylim(0., ax1.get_ylim()[1])
    ax1.set_xlabel(r'Averaged distance $\lambda_{\mathrm{avg}}$')
    ax1.set_ylabel(r'Path length $L$')
    ax1.set_title(r'$\lambda_{\mathrm{avg}}$ -- $L$')
    ax1.legend(loc='best')

    ax2.set_xlabel(r'$\beta$')
    ax2.set_ylabel(r'Fractal dimension $D$')
    ax2.set_title(r'$\beta$ -- $D$')
    ax2.plot(betas, Ds, 'o')

    plt.show()

def variation_of_fd(distance_data, plot=True, save_image=False):
    T = []
    D_T = []
    for frames in sorted(distance_data.keys()):
        T.append(frames)
        # fig, ax1 = plt.subplots()
        betas, Ds = [], []
        for i, f in enumerate(distance_data[frames]):
            fig, ax1 = plt.subplots()
            # beta, D = calc_fractal_dim_for_each_beta(ax1, i, f)
            beta, D = calc_fractal_dim_for_each_beta_manual(ax1, i, f, save_image)
            betas.append(beta)
            Ds.append(D)

        D_T.append(Ds)

        # ax1.set_ylim(0., ax1.get_ylim()[1])
        # ax1.set_xlabel(r'Averaged distance $\lambda_{\mathrm{avg}}$')
        # ax1.set_ylabel(r'Path length $L$')
        # ax1.set_title(r'$\lambda_{\mathrm{avg}}$ -- $L$ (frames=%d)' % frames)
        # ax1.legend(loc='best')

        # if plot:
        #     plt.show()
        # elif save_image:
        #     result_image_path = "results/img/fractal_dim/d_L_frames=%d" % frames
        #     result_image_path += "_" + time.strftime("%y%m%d_%H%M%S")
        #     result_image_path += ".png"
        #     plt.savefig(result_image_path)
        #     plt.close()
        #     print "[saved] " + result_image_path
        # else:
        #     plt.close()

        # fig, ax2 = plt.subplots()
        # ax2.set_xlabel(r'$\beta$')
        # ax2.set_ylabel(r'Fractal dimension $D$')
        # ax2.set_title(r'$\beta$ -- $D$')
        # ax2.plot(betas, Ds, 'o')

        # if plot:
        #     plt.show()
        # elif save_image:
        #     result_image_path = "results/img/fractal_dim/t_D_frames=%d" % frames
        #     result_image_path += "_" + time.strftime("%y%m%d_%H%M%S")
        #     result_image_path += ".png"
        #     plt.savefig(result_image_path)
        #     plt.close()
        #     print "[saved] " + result_image_path
        # else:
        #     plt.close()

    T = np.array(T)
    D = np.array(D_T).T
    betas = np.array(betas)

    # X, Y = np.meshgrid(betas, T)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # ax.plot_wireframe(X, Y, D.T, rstride=1)
    # ax.set_zlim(1., ax.get_zlim()[1])
    # ax.set_title('Fractal dimension')
    # ax.set_xlabel(r'$\beta$')
    # ax.set_ylabel(r'$T$')
    # ax.set_zlabel(r'$D(T)$')
    # plt.show()

    fig, ax = plt.subplots()
    for i, (beta, D_beta) in enumerate(zip(betas, D)):
        color = cm.viridis(float(i) / (len(betas) - 1))
        ax.plot(T, D_beta, ls='', marker='o',
                label=r'$\beta = %2.2f$' % beta,
                color=color,
                markeredgewidth=0.0
                )

    ax.set_xlim(0, ax.get_xlim()[1] + 100)
    ax.set_ylim(1., ax.get_ylim()[1])
    ax.set_xlabel(r'$T$')
    ax.set_ylabel(r'$D(T)$')
    ax.set_title('Fractal dimension')
    ax.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    import time
    from fractal_dim_data import result_data_path, distance_data
    # fractal_dims(result_data_path)
    variation_of_fd(distance_data, plot=False, save_image=True)
