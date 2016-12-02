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
# result_data_path_base = "./results/data/box_counting/2016-11-25/"
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

# fpath = sorted(glob.glob('./results/data/box_counting/*.npz'))
# fpath = sorted(glob.glob('./results/data/box_counting/modified/*.npz'))
# fpath = sorted(glob.glob('./results/data/box_counting/2016-11-19/*.npz'))
# fpath = sorted(glob.glob('./results/data/box_counting/2016-11-25/*.npz'))
# fpath = sorted(glob.glob('./results/data/box_counting/2016-11-26/*.npz'))
# fpath = sorted(glob.glob('./results/data/box_counting/2016-11-28/*.npz'))
fpath = sorted(glob.glob('./results/data/box_counting/2016-12-01/*.npz'))


def plot_Ds():
    fig, ax = plt.subplots()
    D = {}
    for i, result_data_path in enumerate(fpath):
        data = np.load(result_data_path)
        beta = float(data['beta'])
        frames = data['frames']
        Ds = data['Ds']
        alpha = 0.04
        T = (1. / alpha) * np.log(np.arange(frames) / 2. + 1.)
        # filtered = np.where(Ds < 1.)
        # Ds[filtered] = 1.

        if D.has_key(beta):
            D[beta].append(Ds)
        else:
            D[beta] = [Ds]

    betas = sorted(D.keys())

    ## 1) 全てプロット
    # for i, beta in enumerate(betas):
    #     for i, d in D[beta]:
    #         ax.plot(T, d, '.', color=cm.viridis(float(i) / len(betas)))

    ## 2) 指定したbetaのデータをすべてプロット
    # beta = 2.
    # for i, d in enumerate(D[beta]):

    #     color = cm.viridis(float(i) / len(D[beta]))
    #     ax.plot(T[5:], d[5:], '.-', label='data {}'.format(i), color=color)

    ## 3) 指定したbetaの平均とエラー（標準偏差）をプロット
    # beta = 8.
    # D_ave = np.average(np.array(D[beta]), axis=0)
    # D_err = np.std(np.array(D[beta]), axis=0)
    # ax.errorbar(T, D_ave, yerr=D_err, marker='.', ecolor=[0, 0, 0, 0.2])

    ## 4) すべてのbetaごとにプロット

    ## 4.0)
    D_ave = np.array([np.average(np.array(D[k]), axis=0) for k in betas])
    D_err = np.array([np.std(np.array(D[k]), axis=0) for k in betas])

    ## 4.a) フラクタル次元が1以下の部分のデータはすべて無視
    # D_ave = np.ma.array([np.ma.average(
    #     np.ma.array(np.array(D[k]), mask=np.array(D[k])<1.), axis=0)
    #                   for k in betas])
    # D_err = np.ma.array([np.ma.std(
    #     np.ma.array(np.array(D[k]), mask=np.array(D[k])<1.), axis=0)
    #                   for k in betas])

    ## 4.b) フラクタル次元が1以下の部分のデータはすべて1としてプロット
    # D_ave, D_err = [], []
    # for k in betas:
    #     d = np.array(D[k])
    #     d[d < 1] = 1.
    #     D_ave.append(np.average(d, axis=0))
    #     D_err.append(np.std(d, axis=0))

    for i, (beta, d, d_err) in enumerate(zip(betas, D_ave, D_err)):
        color = list(cm.viridis(float(i) / len(betas)))
        label = label=r'$\beta = %2.2f$' % beta

        ## 4.c,d)
        # ax.plot([T[5], T[-1]], [2., 2.], 'k-', lw=1)

        ## 4.c) 平均値のみプロット
        # ax.plot(T, d, '.', label=label, color=color)

        ## 4.d) エラーバーを付けてプロット
        ecolor = color[:-1] + [0.2]
        ax.errorbar(T[5:], d[5:], yerr=d_err[5:], marker='.',
                    label=label, color=color, ecolor=ecolor)

        ## 4.e) 標準偏差の収束をプロットしてみる
        ## 標準偏差は指数関数的に減少
        # ax.semilogy(T, d_err, '.', label=label, color=color)

    ax.legend(loc='best')

    ax.set_title('Fractal dimension')
    ax.set_xlabel(r'$T$')
    # ax.set_ylim((0., 2.5))
    ax.set_ylabel(r'$D(T)$')
    # ax.set_ylabel(r'$\sigma(D(T))$')

    plt.show()

def plot_Ds_3d():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    D = {}
    for i, result_data_path in enumerate(fpath):
        data = np.load(result_data_path)
        beta = float(data['beta'])
        frames = data['frames']
        Ds = data['Ds']
        alpha = 0.04
        T = (1. / alpha) * np.log(np.arange(frames) / 2. + 1.)

        if D.has_key(beta):
            D[beta].append(Ds)
        else:
            D[beta] = [Ds]

    # betas = np.array(betas)
    betas = sorted(D.keys())
    D_ave = np.array([np.average(np.array(D[k]), axis=0) for k in betas])
    D_err = np.array([np.std(np.array(D[k]), axis=0) for k in betas])
    X, Y = np.meshgrid(T, betas)
    ax.plot_wireframe(X, Y, D_ave - D_err, cstride=10, rstride=1, color='g', alpha=0.4)
    ax.plot_wireframe(X, Y, D_ave + D_err, cstride=10, rstride=1, color='r', alpha=0.4)
    ax.plot_wireframe(X, Y, D_ave, cstride=10, rstride=1)
    ax.set_title('Fractal dimension')
    ax.set_xlabel(r'$T$')
    ax.set_ylabel(r'$\beta$')
    ax.set_zlabel(r'$D(T)$')

    plt.show()


if __name__ == '__main__':
    plot_Ds()
    # plot_Ds_3d()
