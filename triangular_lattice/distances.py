#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-08-01

from growing_string import Main
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random


def choose_indexes(_list, num, L):
    """Choose the index pairs whose width is fixed. """
    N = len(_list)
    if N - (2 * L) < num:
        raise StopIteration('list index is smaller than expected (%d), '
                            % (num + 2 * L)
                            + 'given (%d).' % N
                            )
    return sorted(random.sample(_list[L:N - L], num))


if __name__ == '__main__':

    distance_list = []
    path_length = []
    num_strings = 10
    L = 60
    for s in tqdm(range(num_strings)):
        main = Main(Lx=L, Ly=L, size=[3,] * 1, plot=False, frames=1000,
                    strings=[{'id': 1, 'x': L/4, 'y': L/2, 'vec': [0, 4]}])

        num_of_pairs = 300
        len_vec = len(main.strings[0].vec)

        # # 1. 同string内の2点を選ぶ
        # # (1.A) ランダム
        # random_i = np.random.randint(len_vec, size=num_of_pairs)
        # random_j = np.random.randint(len_vec, size=num_of_pairs)

        # (1.B) 等パス長となる2点を同数ずつ抽出
        Lp = range(2, 352)  # stop < 353(= (1000 + 3 - 300) / 2)
        random_i, random_j = [], []
        for lp in Lp:
            random_i.append(np.array(choose_indexes(range(len_vec + 1),
                                                    num_of_pairs, lp)))
            random_j.append(random_i[-1] + lp)

        random_i = np.array(random_i).flatten()
        random_j = np.array(random_j).flatten()

        # 2. 実座標上での距離を算出
        x0, y0 = main.strings[0].pos_x[random_i], main.strings[0].pos_y[random_i]
        x1, y1 = main.strings[0].pos_x[random_j], main.strings[0].pos_y[random_j]
        dx = main.lattice_X[x1, y1] - main.lattice_X[x0, y0]
        dy = main.lattice_Y[x1, y1] - main.lattice_Y[x0, y0]
        distance = np.sqrt(dx**2. + dy**2.)
        # 3. 2点間のベクトル数(=パス長)を計算
        lattices = np.sort(np.array([random_i, random_j]).T)
        lattice_distance = lattices[:, 1] - lattices[:, 0]

        distance_list.append(list(distance))
        path_length.append(list(lattice_distance))

    fig, ax = plt.subplots()
    # # scatter
    # ax.scatter(np.array(distance_list).flatten(),
    #            np.array(path_length).flatten(),
    #            marker='.', s=20)

    # heatmap
    ax.hist2d(np.array(distance_list).flatten(),
              np.array(path_length).flatten(),
              bins=25)

    ax.set_xlabel('Distance')
    ax.set_ylabel('Path length')
    ax.set_title('Path length and distances between two points in the 1D object')
    plt.show()
