#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-08-24

from growing_string import Main
import numpy as np
import matplotlib.pyplot as plt
from surface import get_surface_points, set_labels, get_labeled_position
from Optimize import Optimize_linear


def diecutting_one_cluster(x, y, cutting_size_x, cutting_size_y, x0, y0):
    cluster_indexes = indexes_one_edge(x > x0)                  & \
                      indexes_one_edge(y > y0)                  & \
                      indexes_one_edge(x < (x0 + cutting_size_x)) & \
                      indexes_one_edge(y < (y0 + cutting_size_y))
    return sorted(list(cluster_indexes))

def indexes_one_edge(condition):
    return set(np.argwhere(condition).flatten().tolist())

def eval_subclusters(main, x, y, cutting_size_x, cutting_size_y, x0, y0,
                     plot=False):
    # 長方形領域でクラスターを分割
    cluster_indexes = diecutting_one_cluster(
        x, y,
        cutting_size_x, cutting_size_y,
        x0, y0
    )
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
    if plot:
        fig, ax = plt.subplots()
        pos = list(np.array(main.strings[0].pos).T)
        x = main.lattice_X[pos]
        y = main.lattice_Y[pos]
        ax.plot(x, y, marker='.', color='k', alpha=0.5)
        rect = plt.Rectangle((x0, y0), cutting_size_x, cutting_size_y,
                            facecolor='#f5f5f5')
        plt.gca().add_patch(rect)
        for _sub_cluster in sub_cluster:
            pos = list(np.array(main.strings[0].pos[_sub_cluster]).T)
            x = main.lattice_X[pos]
            y = main.lattice_Y[pos]
            ax.plot(x, y, '.-')
        ax.set_xlim((np.min(main.lattice_X), np.max(main.lattice_X)))
        ax.set_ylim((np.min(main.lattice_Y), np.max(main.lattice_Y)))
        ax.set_title('Strings in rectangular region')
        plt.show()
    # =========================================================================

    return sub_cluster


if __name__ == '__main__':

    # constants
    num_strings = 10
    # steps = 1000
    # L = 60
    steps = 3000
    L = 120
    # steps = 10000
    # L = 240
    # note: stepsとLの大きさに注意


    main = Main(Lx=L, Ly=L, size=[3,] * 1, plot=False, frames=steps,
                strings=[{'id': 1, 'x': L/2, 'y': L/4, 'vec': [0, 4]}]
                )
    s = main.strings[0]

    x = main.lattice_X[np.array(s.pos.T).tolist()]
    y = main.lattice_Y[np.array(s.pos.T).tolist()]
    X = np.average(x)
    Y = np.average(y)

    # # 最大のカットサイズを求める
    position = get_surface_points(main, s)
    label_lattice = set_labels(main, position)
    label_list = label_lattice[position]
    pos = get_labeled_position(main, s, test=False)
    x_surface = main.lattice_X[pos]
    y_surface = main.lattice_Y[pos]
    r_surface = np.sqrt((x_surface - X) ** 2 + (y_surface - Y) ** 2)
    cutting_size_max_width = int(4 * np.min(r_surface) /
                                 (np.sqrt(7) * np.sqrt(3)))
    cutting_size_max_height = cutting_size_max_width * (np.sqrt(3) / 2)
    # print cutting_size_max_width

    X0 = int(X * 2) / 2. - cutting_size_max_width - 0.25
    Y0 = Y - cutting_size_max_height
    # to verify
    # eval_subclusters(main, x, y,
    #                  cutting_size_max_width * 2, cutting_size_max_height * 2,
    #                  X0, Y0, plot=True)

    # 指定した範囲の中に
    #   1)サブクラスターはいくつあるか
    #   2)最大クラスターサイズはいくつか
    num_of_sub_clusters = []
    max_size_of_sub_cluster = []

    cutting_size_xs = np.arange(2, cutting_size_max_width * 2 - 1)
    cutting_size_ys = cutting_size_xs * (np.sqrt(3) / 2)
    cutting_sizes = np.array([cutting_size_xs, cutting_size_ys]).T

    for cutting_size_x, cutting_size_y in cutting_sizes:
        # sample = 3
        sample = 100
        indexes = diecutting_one_cluster(
            x, y,
            cutting_size_max_width * 2 - cutting_size_x,
            cutting_size_max_height * 2 - cutting_size_y,
            X0, Y0
        )
        random_index = list(np.random.choice(indexes, size=sample))
        pos_index = list(main.strings[0].pos[random_index].T)

        _num_of_sub_clusters = []
        _max_size_of_sub_cluster = []

        for x0, y0 in zip(main.lattice_X[pos_index], main.lattice_Y[pos_index]):
            subclusters = eval_subclusters(
                main,
                x, y,
                cutting_size_x, cutting_size_y,
                x0 = int(x0 * 2) / 2. - 0.25,
                y0 = y0 - (np.sqrt(3) / 4),
                plot=False
            )
            _num_of_sub_clusters.append(len(subclusters))
            _max_size_of_sub_cluster.append(max(map(len, subclusters)))

        num_of_sub_clusters.append(np.average(_num_of_sub_clusters))
        max_size_of_sub_cluster.append(np.average(_max_size_of_sub_cluster))

    # print num_of_sub_clusters
    # print max_size_of_sub_cluster
    # =========================================================================

    optimizer = Optimize_linear(
        args=(
            cutting_size_xs[:-1],
            num_of_sub_clusters[:-1]
        ),
        parameters=[1.5, 0.]
    )
    result = optimizer.fitting()
    fig, ax = plt.subplots()
    ax.plot(cutting_size_xs, num_of_sub_clusters, 'o-')
    ax.plot(
        cutting_size_xs[:-1],
        optimizer.fitted(cutting_size_xs[:-1]),
        '-',
        label='a = %f' % result['a']
    )
    ax.legend(loc='best')
    ax.set_title('Strings in rectangular region')
    ax.set_xlabel('Cutting size')
    ax.set_ylabel('Average number of sub-clusters in a rectangular region')
    plt.show()

    max_sub_cluster = max_size_of_sub_cluster / (cutting_size_xs ** 2)
    # optimizer = Optimize_linear(
    #     args=(
    #         cutting_size_xs[:] ** 2,
    #         np.log(max_sub_cluster[:])
    #     ),
    #     parameters=[-1., 0.]
    # )
    # result = optimizer.fitting()
    fig, ax = plt.subplots()
    ax.plot(cutting_size_xs, max_size_of_sub_cluster, 'o-')

    # ax.loglog(cutting_size_xs, max_sub_cluster, 'o-')

    # ax.plot(cutting_size_xs, np.exp(optimizer.fitted(cutting_size_xs)), '-',
    #         label='a = %f' % result['a'])
    # ax.legend(loc='best')
    ax.set_title('Strings in rectangular region')
    ax.set_xlabel('Cutting size')
    ax.set_ylabel('Average of max sub-cluster size in a rectangular region')
    plt.show()

