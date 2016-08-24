#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-08-24

from growing_string import Main
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np
# from tqdm import tqdm
# from multiprocessing import Pool

# constants
num_strings = 10
steps = 1000
L = 60
# steps = 10000
# L = 240
# note: stepsとLの大きさに注意


def diecutting_one_cluster(x, y, cutting_size, x0, y0):
    cluster_indexes = indexes_one_edge(x > x0)                  & \
                      indexes_one_edge(y > y0)                  & \
                      indexes_one_edge(x < (x0 + cutting_size)) & \
                      indexes_one_edge(y < (y0 + cutting_size))
    return list(cluster_indexes)

def indexes_one_edge(condition):
    return set(np.argwhere(condition).flatten().tolist())

def eval_number_of_subclusters(main, x, y, cutting_size, x0, y0):
    # 一辺cutting_sizeの正方形領域でクラスターを分割
    cluster_indexes = diecutting_one_cluster(x, y, cutting_size, x0, y0)
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

    # # 確認 ==================================================================
    # fig, ax = plt.subplots()
    # rect = plt.Rectangle((x0, y0), cutting_size, cutting_size,
    #                      facecolor='#f5f5f5')
    # plt.gca().add_patch(rect)
    # for _sub_cluster in sub_cluster:
    #     pos = list(np.array(main.strings[0].pos[_sub_cluster]).T)
    #     x = main.lattice_X[pos]
    #     y = main.lattice_Y[pos]
    #     ax.plot(x, y, 'o-')
    # ax.set_xlim((np.min(main.lattice_X), np.max(main.lattice_X)))
    # ax.set_ylim((np.min(main.lattice_Y), np.max(main.lattice_Y)))
    # ax.set_title('Strings in rectangular region')
    # plt.show()
    # =========================================================================

    return len(sub_cluster)


if __name__ == '__main__':

    # cutting_size = 20

    main = Main(Lx=L, Ly=L, size=[3,] * 1, plot=False, frames=steps,
                strings=[{'id': 1, 'x': L/2, 'y': L/4, 'vec': [0, 4]}]
                )
    pos_index = list(main.strings[0].pos.T)
    x = main.lattice_X[pos_index]
    y = main.lattice_Y[pos_index]

    cutting_size_max = max(np.max(x) - np.min(x), np.max(y) - np.min(y))
    # cutting_sizes = np.logspace(1, np.log2(cutting_size_max), base=2.)
    cutting_sizes = np.linspace(2, cutting_size_max)
    # print cutting_sizes

    num_of_sub_clusters = []

    for cutting_size in cutting_sizes:
        sample = 1000
        random_index = list(np.random.randint(len(main.strings[0].pos), size=sample))
        pos_index = list(main.strings[0].pos[random_index].T)
        x0s = main.lattice_X[pos_index] - (cutting_size / 2.)
        y0s = main.lattice_Y[pos_index] - (cutting_size / 2.)
        # since dx = 1

        _num_of_sub_clusters = []
        for x0, y0 in zip(x0s, y0s):
            N = eval_number_of_subclusters(main, x, y, cutting_size, x0, y0)
            _num_of_sub_clusters.append(N)

        num_of_sub_clusters.append(np.average(_num_of_sub_clusters))

    print num_of_sub_clusters

    fig, ax = plt.subplots()
    # ax.loglog(cutting_sizes, num_of_sub_clusters, 'o-')
    ax.plot(cutting_sizes, num_of_sub_clusters, 'o-')
    ax.set_title('Strings in rectangular region')
    ax.set_xlabel('Cutting size')
    ax.set_ylabel('Average number of sub-clusters in a rectangular region')
    plt.show()
