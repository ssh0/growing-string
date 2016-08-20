#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-08-01

from growing_string import Main
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


if __name__ == '__main__':

    distance_list = []
    path_length = []
    num_strings = 20
    L = 60
    for s in tqdm(range(num_strings)):
        main = Main(Lx=L, Ly=L, size=[3,] * 1, plot=False, frames=1000,
                    strings=[{'id': 1, 'x': L/2, 'y': L/4, 'vec': [0, 4]}])

        # 1つの線分の長さは1
        # print main.lattice.dy
        # print main.lattice.dx

        # 出来上がったあとのクラスターに着目する ==================================
        # 1. string 内のある点を選ぶ
        # 2. 同じStringに属する別の点をランダムに選択。
        # 3. その点の格子座標を参照。元の点までのベクトル数を数える。

        num_of_pairs = 300
        len_vec = len(main.strings[0].vec)
        # 1つのStringからランダムに2点を選択(index)
        random_i = np.random.randint(len_vec, size=num_of_pairs)
        random_j = np.random.randint(len_vec, size=num_of_pairs)
        x0, y0 = main.strings[0].pos_x[random_i], main.strings[0].pos_y[random_i]
        x1, y1 = main.strings[0].pos_x[random_j], main.strings[0].pos_y[random_j]

        dx = main.lattice_X[x1, y1] - main.lattice_X[x0, y0]
        dy = main.lattice_Y[x1, y1] - main.lattice_Y[x0, y0]
        distance = np.sqrt(dx**2. + dy**2.)
        lattices = np.sort(np.array([random_i, random_j]).T)
        lattice_distance = lattices[:, 1] - lattices[:, 0]

        distance_list.append(list(distance))
        path_length.append(list(lattice_distance))

    fig, ax = plt.subplots()
    # scatter
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
