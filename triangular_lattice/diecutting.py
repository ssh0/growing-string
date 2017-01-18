#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-08-24

from growing_string import Main
from surface import get_surface_points, set_labels, get_labeled_position
from optimize import Optimize_linear
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import itertools


class DieCutting(object):
    def __init__(self, L=60, frames=1000, sample=3, beta=2., c=0.5,
                 plot=False):
        self.L = L
        self.frames = frames
        self.sample = sample
        self.beta = beta
        self.weight_const = c
        self.plot_for_veirification = plot

    def init(self):
        self.main = Main(
            Lx=self.L,
            Ly=self.L,
            size=[3,] * 1,
            plot=False,
            frames=self.frames,
            beta=self.beta,
            weight_const=self.weight_const,
            strings=[{'id': 1, 'x': self.L/4, 'y': self.L/2, 'vec': [0, 4]}]
        )

        self.s = self.main.strings[0]

        self.lattice_X = self.main.lattice_X
        self.lattice_Y = self.main.lattice_Y
        self.x = self.lattice_X[np.array(self.s.pos.T).tolist()]
        self.y = self.lattice_Y[np.array(self.s.pos.T).tolist()]

    def start(self, result_set, visualize=True):
        self.init()
        self.get_cutting_sizes()
        self.resultset = result_set
        self.res = self.get_results_each_subclusters(self.resultset)
        # print self.res
        if visualize:
            self.visualize_results()

    def get_cutting_sizes(self):
        """Create the cutting size list for simulation

        self.X0: x coordinates of the left buttom corner
        self.Y0: y coordinates of the left buttom corner
        self.cutting_size_max_width: max width of the cutting size
        self.cutting_size_max_height: max height of the cutting size
        self.cutting_size_xs: cutting size list
        self.cutting_size_ys: cutting size list
        self.cutting_sizes: ndarray [[cutting_size_xs[0], cutting_size_ys[0]],
                                     [cutting_size_xs[1], cutting_size_ys[1]],
                                         ...
                                    ]

        In this funciton, cutting size is determined by which the whole region
        is in the cluster.
        """
        X = np.average(self.x)
        Y = np.average(self.y)
        # surfaceモジュールで最外部の占有点の座標を取得(x_surface, y_surface)
        position = get_surface_points(self.main, self.s.pos)
        label_lattice = set_labels(self.main, position)
        label_list = label_lattice[position]
        pos = get_labeled_position(self.main, self.s.pos, test=False)
        x_surface = self.lattice_X[pos]
        y_surface = self.lattice_Y[pos]
        r_surface = np.sqrt((x_surface - X) ** 2 + (y_surface - Y) ** 2)

        max_half_width = int(4 * np.min(r_surface) / (np.sqrt(7) * np.sqrt(3)))
        max_half_height = max_half_width * (np.sqrt(3) / 2)

        self.X0 = int(X * 2) / 2. - max_half_width - 0.25
        self.Y0 = Y - max_half_height

        self.cutting_size_max_width = 2 * max_half_width
        self.cutting_size_max_height = 2 * max_half_height

        self.cutting_size_xs = np.arange(2, self.cutting_size_max_width - 1)
        self.cutting_size_ys = self.cutting_size_xs * (np.sqrt(3) / 2)
        self.cutting_sizes = np.array([self.cutting_size_xs,
                                       self.cutting_size_ys]).T

        # to verify
        # self.eval_subclusters(
        #     self.cutting_size_max_width,
        #     self.cutting_size_max_height,
        #     self.X0,
        #     self.Y0
        # )
        return

    def diecutting_one_cluster(self, width, height, x0, y0):
        cluster_indexes = self.indexes_one_edge(self.x > x0) & \
            self.indexes_one_edge(self.y > y0)               & \
            self.indexes_one_edge(self.x < (x0 + width))     & \
            self.indexes_one_edge(self.y < (y0 + height))
        return sorted(list(cluster_indexes))

    def indexes_one_edge(self, condition):
        return set(np.argwhere(condition).flatten().tolist())

    def eval_subclusters(self, width, height, x0, y0):
        # 長方形領域でクラスターを分割
        cluster_indexes = self.diecutting_one_cluster(width, height, x0, y0)
        # print cluster_indexes

        # 領域内で独立したクラスターは何個?
        sub_cluster = []
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
            rect = plt.Rectangle((x0, y0), width, height,
                                facecolor='#f5f5f5', edgecolor='#ff0000', lw=2)
            plt.gca().add_patch(rect)
            for _sub_cluster in sub_cluster:
                pos = list(np.array(self.s.pos[_sub_cluster]).T)
                x = self.lattice_X[pos]
                y = self.lattice_Y[pos]
                ax.plot(x, y, '.-')
            ax.set_xlim((np.min(self.lattice_X), np.max(self.lattice_X)))
            ax.set_ylim((np.min(self.lattice_Y), np.max(self.lattice_Y)))
            ax.set_title('Strings in rectangular region')
            ax.set_aspect('equal')
            plt.show()
        # =====================================================================

        return sub_cluster

    def get_results_each_subclusters(self, sets):
        res = {k: [] for k in sets.keys()}
        for width, height in self.cutting_sizes:
            indexes = self.diecutting_one_cluster(
                self.cutting_size_max_width - width,
                self.cutting_size_max_height - height,
                self.X0,
                self.Y0
            )
            random_index = list(np.random.choice(indexes, size=self.sample))
            pos = list(self.s.pos[random_index].T)
            _res = {k: [] for k in sets.keys()}
            for x0, y0 in zip(self.lattice_X[pos], self.lattice_Y[pos]):
                subclusters = self.eval_subclusters(
                    width,
                    height,
                    x0=int(x0 * 2) / 2. - 0.25,
                    y0=y0 - (np.sqrt(3) / 4),
                )
                for k in sets.keys():
                    _res[k].append(sets[k]['_func'](subclusters))
            for k in sets.keys():
                res[k].append(sets[k]['func'](_res[k]))
        return res

    def visualize_results(self):
        for k in self.res.keys():
            getattr(self, 'visualize_' + k)(self)


def visualize_num_of_sub_clusters(self):
    fig, ax = plt.subplots()

    num_of_sub_clusters = self.res['num_of_sub_clusters']
    optimizer = Optimize_linear(
        args=(
            self.cutting_size_xs[:-1],
            num_of_sub_clusters[:-1]
        ),
        parameters=[1.5, 0.]
    )
    result = optimizer.fitting()

    ax.plot(self.cutting_size_xs, num_of_sub_clusters, 'o-')
    ax.plot(
        self.cutting_size_xs[:-1],
        optimizer.fitted(self.cutting_size_xs[:-1]),
        '-',
        label='a = %f' % result['a']
    )
    ax.legend(loc='best')
    ax.set_title('Strings in rectangular region')
    ax.set_xlabel('Cutting size')
    ax.set_ylabel('Average number of sub-clusters in a rectangular region')
    plt.show()

def visualize_max_size_of_sub_cluster(self):
    max_size_of_sub_cluster = self.res['max_size_of_sub_cluster']
    optimizer = Optimize_linear(
        args=(
            self.cutting_size_xs[:],
            max_size_of_sub_cluster[:]
        ),
        parameters=[-1., 0.]
    )
    result = optimizer.fitting()
    fig, ax = plt.subplots()
    ax.plot(self.cutting_size_xs, max_size_of_sub_cluster, 'o-')

    # ax.loglog(self.cutting_size_xs, max_size_of_sub_cluster, 'o-')

    ax.plot(self.cutting_size_xs, optimizer.fitted(self.cutting_size_xs), '-',
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
    main = DieCutting(
        L=60,
        frames=1000,
        # L=120,
        # frames=3000,
        sample=3,
        beta=0.,
        plot=True)

    main.visualize_num_of_sub_clusters = visualize_num_of_sub_clusters
    main.visualize_size_dist_of_sub_clusters = visualize_size_dist_of_sub_clusters
    main.visualize_max_size_of_sub_cluster = visualize_max_size_of_sub_cluster

    result_set = {
        'num_of_sub_clusters': {
            '_func': len,
            'func': np.average
        },
        'size_dist_of_sub_clusters': {
            '_func': lambda arr: np.bincount(map(len, arr)),
            'func': lambda arr: map(sum, itertools.izip_longest(*arr, fillvalue=0))
        },
        # 'max_size_of_sub_cluster': {
        #     '_func': lambda arr: max(map(len, arr)),
        #     'func': np.average
        # }
    }

    main.start(result_set, visualize=True)

