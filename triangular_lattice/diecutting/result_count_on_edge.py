#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-12-16


import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.cm as cm
import numpy as np
import set_data_path


class Visualizer(object):
    def __init__(self, subjects):
        self.data_path_list = set_data_path.data_path
        if len(subjects) != 0:
            for subject in subjects:
                getattr(self, 'result_' + subject)()

    def load_data(self, _path):
        data = np.load(_path)
        beta = data['beta']
        try:
            size_dist_ave = data['size_dist_ave']
            if len(size_dist_ave) == 0:
                raise KeyError
            return self.load_data_averaged(_path)
        except KeyError:
            pass

        num_of_strings = data['num_of_strings']
        frames = data['frames']
        Ls = data['Ls'].astype(np.float)
        # Ls = (3 * Ls * (Ls + 1) + 1)
        size_dist = data['size_dist']

        N0 = np.array([l[1] for l in size_dist], dtype=np.float) / num_of_strings
        n0 = N0[1:]
        S = np.array([np.sum(l) for l in size_dist], dtype=np.float) / num_of_strings
        n1 = (S[1:] - n0) * 2.

        N = []
        for l in size_dist:
            dot = np.dot(np.arange(len(l)), np.array(l).T)
            N.append(dot)
        # N = np.array([np.dot(np.arange(len(l)), np.array(l).T) for l in size_dist])
        N_all = 3. * Ls * (Ls + 1.) + 1
        N = np.array(N, dtype=np.float) / num_of_strings
        N_minus = N_all - N
        N_minus_rate = N_minus / N_all
        n_minus = N_minus[1:] - N_minus[:-1]
        n1_ave = n1 / np.sum(n1)
        n2 = (6 * Ls[1:]) - (n0 + n1 + n_minus)

        self.beta = beta
        self.num_of_strings = num_of_strings
        self.frames = frames
        self.Ls = Ls
        self.N = N
        self.N_minus = N_minus
        self.N_minus_rate = N_minus_rate
        self.S = S
        self.n0 = n0
        self.n1 = n1
        self.n2 = n2
        self.n_minus = n_minus
        self.n1_ave = n1_ave

    def load_data_averaged(self, _path):
        data = np.load(_path)
        beta = data['beta']
        num_of_strings = data['num_of_strings']
        frames = data['frames']
        Ls = data['Ls'].astype(np.float)
        # Ls = (3 * Ls * (Ls + 1) + 1)
        # size_dist = data['size_dist']
        size_dist_ave = data['size_dist_ave']

        N0 = np.array([l[1] for l in size_dist_ave], dtype=np.float)
        n0 = N0[1:]
        S = np.array([np.sum(l) for l in size_dist_ave], dtype=np.float)
        n1 = (S[1:] - n0) * 2.

        N = []
        for l in size_dist_ave:
            dot = np.dot(np.arange(len(l)), np.array(l).T)
            N.append(dot)
        # N = np.array([np.dot(np.arange(len(l)), np.array(l).T) for l in size_dist_ave])
        N_all = 3. * Ls * (Ls + 1.) + 1
        N = np.array(N, dtype=np.float)
        N_minus = N_all - N
        N_minus_rate = N_minus / N_all
        n_minus = N_minus[1:] - N_minus[:-1]
        n1_ave = n1 / np.sum(n1)
        n2 = (6 * Ls[1:]) - (n0 + n1 + n_minus)

        self.beta = beta
        self.num_of_strings = num_of_strings
        self.frames = frames
        self.Ls = Ls
        self.N = N
        self.N_all = N_all
        self.N_minus = N_minus
        self.N_minus_rate = N_minus_rate
        self.S = S
        self.n_all = 6 * Ls[1:]
        self.n0 = n0
        self.n1 = n1
        self.n2 = n2
        self.n_minus = n_minus
        self.n1_ave = n1_ave

    def result_N(self):
        fig, ax = plt.subplots()
        for i, result_data_path in enumerate(self.data_path_list):
            self.load_data(result_data_path)
            ax.plot(self.Ls[1:], self.N[1:], '.',
                    label=r'$\beta = %2.2f$' % self.beta,
                    color=cm.viridis(float(i) / len(self.data_path_list)))
        ax.legend(loc='best')
        ax.set_title('Occupied points in the cutting region' +
                    ' (sample: {})'.format(self.num_of_strings))
        ax.set_xlabel(r'Cutting size $L$')
        ax.set_ylabel(r'$N$')
        plt.show()

    def result_N_minus_rate(self):
        fig, ax = plt.subplots()
        for i, result_data_path in enumerate(self.data_path_list):
            self.load_data(result_data_path)
            ax.plot(self.Ls[1:], self.N_minus_rate[1:], '.',
                    label=r'$\beta = %2.2f$' % self.beta,
                    color=cm.viridis(float(i) / len(self.data_path_list)))
        ax.legend(loc='best')
        ax.set_title('The rate of not occupied site in all N' +
                    ' (sample: {})'.format(self.num_of_strings))
        ax.set_xlabel(r'Cutting size $L$')
        ax.set_ylabel(r'$N_{-1} / N_{\mathrm{all}}$')
        plt.show()

    def result_n0(self):
        fig, ax = plt.subplots()
        for i, result_data_path in enumerate(self.data_path_list):
            self.load_data(result_data_path)
            ax.plot(self.Ls[1:], self.n0, '.',
                    label=r'$\beta = %2.2f$' % self.beta,
                    color=cm.viridis(float(i) / len(self.data_path_list)))
        ax.legend(loc='best')
        ax.set_title('Averaged number of the sites which is the only member of \
                     a subcluster on the cutting edges.' +
                    ' (sample: {})'.format(self.num_of_strings))
        ax.set_xlabel(r'Cutting size $L$')
        ax.set_ylabel(r'$n_{0}$')
        plt.show()

    def result_n1(self):
        fig, ax = plt.subplots()
        for i, result_data_path in enumerate(self.data_path_list):
            self.load_data(result_data_path)
            ax.plot(self.Ls[1:], self.n1, '.',
                    label=r'$\beta = %2.2f$' % self.beta,
                    color=cm.viridis(float(i) / len(self.data_path_list)))
        ax.legend(loc='best')
        ax.set_title('Averaged number of the sites which is connected to a \
                    existing subcluster on the cutting edges.' +
                    ' (sample: {})'.format(self.num_of_strings))
        ax.set_xlabel(r'Cutting size $L$')
        ax.set_ylabel(r'$n_{1}$')
        plt.show()

    def result_n2(self):
        fig, ax = plt.subplots()
        for i, result_data_path in enumerate(self.data_path_list):
            self.load_data(result_data_path)
            ax.plot(self.Ls[1:], self.n2, '.',
                    label=r'$\beta = %2.2f$' % self.beta,
                    color=cm.viridis(float(i) / len(self.data_path_list)))
        ax.legend(loc='best')
        ax.set_title('Averaged number of the sites on the cutting edges which \
                    is connected to two neighbors.' + 
                    ' (sample: {})'.format(self.num_of_strings))
        ax.set_xlabel(r'Cutting size $L$')
        ax.set_ylabel(r'$n_{2}$')
        plt.show()

    def result_n_minus(self):
        fig, ax = plt.subplots()
        for i, result_data_path in enumerate(self.data_path_list):
            self.load_data(result_data_path)
            ax.plot(self.Ls[1:], self.n_minus, '.',
                    label=r'$\beta = %2.2f$' % self.beta,
                    color=cm.viridis(float(i) / len(self.data_path_list)))
        ax.legend(loc='best')
        ax.set_title('Averaged number of the sites which is not occupied on \
                     the cutting edges.' + 
                    ' (sample: {})'.format(self.num_of_strings))
        ax.set_xlabel(r'Cutting size $L$')
        ax.set_ylabel(r'$n_{-1}$')
        plt.show()

    def result_S(self):
        fig, ax = plt.subplots()
        for i, result_data_path in enumerate(self.data_path_list):
            self.load_data(result_data_path)
            ax.plot(self.Ls[1:], self.S[1:], '.',
                    label=r'$\beta = %2.2f$' % self.beta,
                    color=cm.viridis(float(i) / len(self.data_path_list)))
        ax.legend(loc='best')
        ax.set_ylim([0, ax.get_ylim()[1]])
        ax.set_title('Averaged number of the subclusters in the cutted region.'
                     + ' (sample: {})'.format(self.num_of_strings))
        ax.set_xlabel(r'Cutting size $L$')
        ax.set_ylabel(r'$S$')
        plt.show()

    def result_S_rate(self):
        fig, ax = plt.subplots()
        for i, result_data_path in enumerate(self.data_path_list):
            self.load_data(result_data_path)
            # ax.plot(self.Ls[1:], self.S[1:] / np.sum(self.S[1:]), '.',
            # ax.plot(self.Ls[1:], self.S[1:] / self.n_all, '.',
            ax.plot(self.Ls[1:], self.S[1:] / self.N[1:], '.',
                    label=r'$\beta = %2.2f$' % self.beta,
                    color=cm.viridis(float(i) / len(self.data_path_list)))
        ax.legend(loc='best')
        ax.set_ylim([0, ax.get_ylim()[1]])
        ax.set_title('Averaged number of the subclusters in the cutted region'
                     + ' (normalized)'
                     + ' (sample: {})'.format(self.num_of_strings))
        ax.set_xlabel(r'Cutting size $L$')
        ax.set_ylabel(r'$S$')
        plt.show()


if __name__ == '__main__':
    # subject: 'N', 'N_minus_rate', 'n0', 'n1', 'n2', 'n_minus', 'S'
    main = Visualizer(
        [
            # 'N',
            # 'N_minus_rate',
            # 'n0',
            # 'n1',
            # 'n2',
            # 'n_minus',
            'S',
            # 'S_rate'
        ]
    )
