#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-10-16

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from optimize import Optimize_linear
import itertools
from diecutting import DieCutting


class DieCuttingHexagonal(DieCutting):
    def __init__(self, params):
        parent = DieCutting(**params)
        self.__dict__ = parent.__dict__

    def start(self, result_set, visualize=True):
        self.init()
        self.get_cutting_sizes()
        self.inside_of_hex = [[self.X0, self.Y0], ]
        self.resultset = result_set
        self.res = self.get_results_each_subclusters(self.resultset)
        if visualize:
            self.visualize_results()

    def get_circumfences_of_hexagonal(self, x0, y0, L):
        center = np.array([x0, y0])
        region0 = np.array([[L - i, i] for i in range(L)])
        region1 = np.array([[-i, L] for i in range(L)])
        region2 = np.array([[-L, L - i] for i in range(L)])
        region3 = - region0
        region4 = - region1
        region5 = - region2
        stack = tuple([locals()['region{}'.format(i)] for i in range(6)])
        region = (np.vstack(stack) + center) % self.L
        return region.tolist()

    def eval_subclusters(self):
        cluster_indexes = []
        for i, (lx, ly) in enumerate(self.s.pos):
            if [lx, ly] in self.inside_of_hex:
                cluster_indexes.append(i)

        # 領域内で独立したクラスターは何個?
        sub_cluster = []
        if len(cluster_indexes) == 0:
            return []

        _sub_cluster = [cluster_indexes[0], ]
        for i in cluster_indexes[1:]:
            if i == _sub_cluster[-1] + 1:
                _sub_cluster.append(i)
            else:
                sub_cluster.append(_sub_cluster)
                _sub_cluster = [i]
        sub_cluster.append(_sub_cluster)
        # print sub_cluster
        # print len(sub_cluster)

        # # 確認 ==============================================================
        if self.plot_for_veirification:
            fig, ax = plt.subplots()
            pos = list(np.array(self.s.pos).T)
            x = self.lattice_X[pos]
            y = self.lattice_Y[pos]
            ax.plot(x, y, marker='.', color='k', alpha=0.5)
            for _sub_cluster in sub_cluster:
                pos = list(np.array(self.s.pos[_sub_cluster]).T)
                x = self.lattice_X[pos]
                y = self.lattice_Y[pos]
                ax.plot(x, y, '.-')
            ax.set_xlim((np.min(self.lattice_X), np.max(self.lattice_X)))
            ax.set_ylim((np.min(self.lattice_Y), np.max(self.lattice_Y)))
            ax.set_title('Strings in hexagonal region')
            ax.set_aspect('equal')
            plt.show()
        # =====================================================================

        return sub_cluster

    def get_cutting_sizes(self):
        """Create the cutting size list for simulation

        self.X0: x lattice coordinates of the center
        self.Y0: y lattice coordinates of the center
        self.cutting_size_max: max cutting size
        self.cutting_sizes: cutting size list

        In this funciton, cutting size is determined by cluster size
        """
        X = np.average(self.x)
        Y = np.average(self.y)
        r = np.sqrt((self.x - X) ** 2 + (self.y - Y) ** 2)
        L_max = min(int(np.max(r)) + 1, self.L / 2)
        self.Y0 = np.argmin(abs(self.lattice_Y[0] - Y))
        self.X0 = np.argmin(abs(self.lattice_X[:, self.Y0] - X))

        self.cutting_sizes = np.arange(1, L_max + 1)
        return

    def diecutting_one_cluster(self, x0, y0, L):
        self.inside_of_hex += self.get_circumfences_of_hexagonal(x0, y0, L)

    def get_results_each_subclusters(self, sets):
        res = {k: [] for k in sets.keys()}

        for L in self.cutting_sizes:  # L must be sorted
            self.diecutting_one_cluster(self.X0, self.Y0, L)
            subclusters = self.eval_subclusters()
            _res = {k: [] for k in sets.keys()}
            for k in sets.keys():
                _res[k].append(sets[k]['_func'](subclusters))

            for k in sets.keys():
                res[k].append(sets[k]['func'](_res[k]))
        return res


def visualize_num_of_sub_clusters(self):
    fig, ax = plt.subplots()

    num_of_sub_clusters = self.res['num_of_sub_clusters']
    # optimizer = Optimize_linear(
    #     args=(
    #         self.cutting_sizes[:-1],
    #         num_of_sub_clusters[:-1]
    #     ),
    #     parameters=[1.5, 0.]
    # )
    # result = optimizer.fitting()

    ax.plot(self.cutting_sizes, num_of_sub_clusters, 'o-')
    # ax.plot(
    #     self.cutting_sizes[:-1],
    #     optimizer.fitted(self.cutting_sizes[:-1]),
    #     '-',
    #     label='a = %f' % result['a']
    # )
    # ax.legend(loc='best')
    ax.set_title('Strings in rectangular region')
    ax.set_xlabel('Cutting size')
    ax.set_ylabel('Average number of sub-clusters in a rectangular region')
    plt.show()

def visualize_max_size_of_sub_cluster(self):
    max_size_of_sub_cluster = self.res['max_size_of_sub_cluster']
    optimizer = Optimize_linear(
        args=(
            self.cutting_sizes[:],
            max_size_of_sub_cluster[:]
        ),
        parameters=[-1., 0.]
    )
    result = optimizer.fitting()
    fig, ax = plt.subplots()
    ax.plot(self.cutting_sizes, max_size_of_sub_cluster, 'o-')

    # ax.loglog(self.cutting_sizes, max_size_of_sub_cluster, 'o-')

    ax.plot(self.cutting_sizes, optimizer.fitted(self.cutting_sizes), '-',
            label='a = %f' % result['a'])
    ax.legend(loc='best')
    ax.set_title('Strings in rectangular region')
    ax.set_xlabel('Cutting size')
    ax.set_ylabel('Average of max sub-cluster size in a rectangular region')
    plt.show()

def visualize_size_dist_of_sub_clusters(self):
    lst = self.res['size_dist_of_sub_clusters']

    n = max(map(len, lst))
    res = np.zeros([len(lst), n])
    for i, row in enumerate(lst):
        for j, val in enumerate(row):
            res[i][j] = val

    L = len(res)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(range(len(res[0])), range(L))
    ax.plot_wireframe(X, Y, res, rstride=1)
    ax.set_title("Histogram of the size of subclusters in the region with $L$")
    ax.set_ylim(0, L)
    ax.set_xlabel("Size of a subcluster")
    ax.set_ylabel("$L$")
    ax.set_zlabel("Freq")
    plt.show()


if __name__ == '__main__':
    # === Plot one result ===
    params = {
        'L': 100,
        'frames': 2000,
        'beta': 0.,
        'plot': True
    }
    main = DieCuttingHexagonal(params)
    main.visualize_num_of_sub_clusters = visualize_num_of_sub_clusters
    main.visualize_size_dist_of_sub_clusters = visualize_size_dist_of_sub_clusters
    main.visualize_max_size_of_sub_cluster = visualize_max_size_of_sub_cluster
    result_set = {
        'num_of_sub_clusters': {
            '_func': len,
            'func': np.average
        },
        # 'size_dist_of_sub_clusters': {
        #     '_func': lambda arr: np.bincount(map(len, arr)),
        #     'func': lambda arr: map(sum, itertools.izip_longest(*arr, fillvalue=0))
        # },
        'max_size_of_sub_cluster': {
            '_func': lambda arr: max(map(len, arr)),
            'func': np.average
        }
    }
    main.start(result_set, visualize=True)
