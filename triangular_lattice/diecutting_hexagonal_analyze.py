#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-10-16

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.cm as cm
import numpy as np
import itertools
from scipy.optimize import curve_fit
from scipy.stats import gamma


if __name__ == '__main__':

    result_data_path_base = "./results/data/diecutting/"
    fn = [
        # "beta=0.00_161016_190044.npz",
        # "beta=0.10_161016_190056.npz",
        # "beta=0.20_161016_190105.npz",
        # "beta=0.30_161016_190111.npz",
        # "beta=0.40_161016_190125.npz",
        # "beta=0.50_161016_190158.npz",
        # "beta=0.60_161016_190207.npz",
        # "beta=0.70_161016_190217.npz",
        # "beta=0.80_161016_190229.npz",
        # "beta=0.90_161016_190236.npz",
        # "beta=1.00_161016_190242.npz",
        # "beta=2.00_161016_195423.npz",
        # "beta=3.00_161016_195439.npz",
        # "beta=4.00_161016_195447.npz",
        # "beta=5.00_161016_195452.npz",
        # "beta=6.00_161016_195526.npz",
        # "beta=0.00_161111_132832.npz",
        # "beta=0.00_161111_141810.npz",  # <- beta = 0.001
        # "beta=1.00_161111_132834.npz",
        # "beta=2.00_161111_132842.npz",
        # "beta=3.00_161111_132849.npz",
        # "beta=4.00_161111_132858.npz",
        # "beta=5.00_161111_132907.npz",
        # "beta=6.00_161111_132916.npz",
        # "beta=0.00_161111_143949.npz",
        # "beta=1.00_161111_144002.npz",
        # "beta=2.00_161111_144011.npz",
        # "beta=3.00_161111_144019.npz",
        # "beta=4.00_161111_144025.npz",
        # "beta=5.00_161111_144032.npz",
        # "beta=6.00_161111_144038.npz",

        # size_dist
        "beta=0.00_161114_213905.npz",
        "beta=1.00_161114_213913.npz",
        "beta=2.00_161114_213919.npz",
        "beta=3.00_161114_213927.npz",
        "beta=4.00_161114_213934.npz",
        "beta=5.00_161114_213940.npz",
        "beta=6.00_161114_213946.npz",
        "beta=7.00_161115_150237.npz",
        "beta=8.00_161115_150246.npz",
        "beta=9.00_161115_150252.npz",
        "beta=10.00_161115_150258.npz",
    ]


    def result_N_sub():
        fig, ax = plt.subplots()
        for result_data_path in [result_data_path_base + f for f in fn]:
            data = np.load(result_data_path)
            beta = data['beta']
            num_of_strings = data['num_of_strings']
            L = data['L']
            frames = data['frames']
            Ls = data['Ls'].astype(np.float)
            N_sub = data['N_sub']

            # M = N_sub / 3 * Ls * (Ls + 1) + 1
            M = N_sub
            M_ave = M / np.sum(M)
            popt = curve_fit(gamma.pdf, xdata=Ls, ydata=M_ave, p0=[2.5, -5., 30])[0]
            print beta, popt
            ax.plot(Ls, M_ave, '.-', label=r'$\beta = %2.2f$' % beta)
            x = np.linspace(1., max(Ls), num=5*max(Ls))
            ax.plot(x, gamma.pdf(x, a=popt[0], loc=popt[1], scale=popt[2]),
                    '-', label=r'fitted $\beta = %2.2f$' % beta)
        ax.legend(loc='best')
        ax.set_ylim((0., 0.1))
        ax.set_title('Strings in hexagonal region' +
                    ' (sample: {})'.format(num_of_strings))
        ax.set_xlabel(r'Cutting size $L$')
        ax.set_ylabel('Average number of the sub-clusters in the hexagonal region of size L')

        plt.show()

    def result_size_dist_3d():
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        fpath = [result_data_path_base + f for f in fn]
        for i, result_data_path in enumerate(fpath):
            data = np.load(result_data_path)
            beta = data['beta']
            num_of_strings = data['num_of_strings']
            L = data['L']
            frames = data['frames']
            Ls = data['Ls'].astype(np.float)
            size_dist = data['size_dist']
            X, Y = np.meshgrid(Ls, range(len(size_dist[0])))
            ax.plot_wireframe(X, Y, size_dist.T, rstride=1, alpha=0.3,
                            color=cm.gnuplot(float(i) / len(fpath)))

        ax.set_title("Histogram of the size of subclusters in the region with $L$")
        ax.set_xlabel("$L$")
        ax.set_ylabel("Size of a subcluster")
        ax.set_zlabel("Freq")
        plt.show()

    def result_size_dist_pcolor():
        fig, ax = plt.subplots()
        fpath = [result_data_path_base + f for f in fn]
        for i, result_data_path in enumerate(fpath):
            data = np.load(result_data_path)
            beta = data['beta']
            num_of_strings = data['num_of_strings']
            L = data['L']
            frames = data['frames']
            Ls = data['Ls'].astype(np.float)
            size_dist = data['size_dist']
            size_dist = size_dist / np.max(size_dist)
            # size_dist = np.array([_s / np.sum(_s) for _s in size_dist])
            # size_dist = np.array([_s / np.max(_s) for _s in size_dist])
            X, Y = np.meshgrid(Ls, range(len(size_dist[0])))
            ax.pcolor(X, Y, size_dist.T)

        ax.set_title("Histogram of the size of subclusters in the region with $L$")
        ax.set_xlim(0, max(Ls))
        ax.set_ylim(0, size_dist.shape[1])
        # ax.set_ylim(0, size_dist.shape[0] * 2)
        ax.set_xlabel("$L$")
        ax.set_ylabel("Size of a subcluster")
        ax.set_aspect('equal')
        plt.show()

    def result_length_of_sub_dist():
        fig, ax = plt.subplots()
        fpath = [result_data_path_base + f for f in fn]
        for i, result_data_path in enumerate(fpath):
            data = np.load(result_data_path)
            beta = data['beta']
            num_of_strings = data['num_of_strings']
            L = data['L']
            frames = data['frames']
            Ls = data['Ls'].astype(np.float)
            size_dist = data['size_dist']
            N = np.sum(size_dist, axis=0)
            N_freq = N / np.sum(N)
            X = np.arange(1, len(N)+1)
            # ax.semilogy(X, N_freq, '.', label=r'$\beta = %2.2f$' % beta,
            #           color=cm.viridis(float(i) / len(fpath)))
            ax.semilogy(X[::2], N_freq[::2], '.', label=r'$\beta = %2.2f$' % beta,
                      color=cm.viridis(float(i) / len(fpath)))
            # ax.semilogy(X[1::2], N_freq[1::2], '.', label=r'$\beta = %2.2f$' % beta,
            #           color=cm.viridis(float(i) / len(fpath)))

        ax.legend(loc='best')
        ax.set_title("Appearance frequency of the subcluster of size $N$")
        ax.set_xlabel("$N$")
        ax.set_ylabel("Freq")
        plt.show()

    def result_N_ave():
        fig, ax = plt.subplots()
        fpath = [result_data_path_base + f for f in fn]
        for result_data_path in [result_data_path_base + f for f in fn]:
            data = np.load(result_data_path)
            beta = data['beta']
            num_of_strings = data['num_of_strings']
            L = data['L']
            frames = data['frames']
            Ls = data['Ls'].astype(np.float)
            # Ls = (3 * Ls * (Ls + 1) + 1)
            size_dist = data['size_dist']

            # M = np.array([np.average(np.arange(len(l)), weights=l) for l in size_dist])
            # M = np.array([np.sum(l) / np.sum(size_dist) for l in size_dist])
            M = np.array([np.sum(l) / ((3. * (i + 1) * (i + 2) + 1) * num_of_strings) for i, l in enumerate(size_dist)])

            ax.semilogx(Ls, M, '.-', label=r'$\beta = %2.2f$' % beta)

            popt = curve_fit(gamma.pdf, xdata=Ls, ydata=M, p0=[2., -2.5, 10])[0]
            print beta, popt
            x = np.logspace(np.log(min(Ls)), np.log(max(Ls)), num=5*max(Ls))
            ax.semilogx(x, gamma.pdf(x, a=popt[0], loc=popt[1], scale=popt[2]),
                    '-', label=r'fitted $\beta = %2.2f$' % beta)

        ax.legend(loc='best')
        # ax.set_ylim((0., 0.1))
        ax.set_title('Strings in hexagonal region' +
                    ' (sample: {})'.format(num_of_strings))
        ax.set_xlabel(r'Cutting size $L$')
        ax.set_ylabel('Average number of the sub-clusters in the hexagonal region of size L')

        plt.show()

    ## for N_sub ===========
    # result_N_sub()

    ## for size_dist =======
    # result_size_dist_3d()
    # result_size_dist_pcolor()  # dataは1つにする
    result_length_of_sub_dist()
    # result_N_ave()

