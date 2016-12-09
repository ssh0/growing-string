#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-08-01

from growing_string import Main
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time
import random
import save_data as sd


def choose_indexes(_list, num, L):
    """Choose the index pairs whose width is fixed. """
    N = len(_list)
    if N - (2 * L) < num:
        raise StopIteration('list index is smaller than expected (%d), '
                            % (num + 2 * L)
                            + 'given (%d).' % N
                            )
    return sorted(random.sample(_list[L:N - L], num))

def get_path_length_and_distances(beta, num_of_strings, L, frames, num_of_pairs=300):
    main = Main(Lx=L, Ly=L, size=[3,] * 1, plot=False, frames=frames, beta=beta,
                strings=[{'id': 1, 'x': L/4, 'y': L/2, 'vec': [0, 4]}])
    len_vec = len(main.strings[0].vec)

    # # 1. 同string内の2点を選ぶ
    # # (1.A) ランダム
    # random_i = np.random.randint(len_vec, size=num_of_pairs)
    # random_j = np.random.randint(len_vec, size=num_of_pairs)

    # (1.B) 等パス長となる2点を同数ずつ抽出
    Lp = range(2, (len_vec - num_of_pairs) / 2)
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
    return list(distance), list(lattice_distance)

def execute_simulation_for_one_beta(beta, num_of_strings, L, frames, 
                                    num_of_pairs, plot=True,
                                    save_image=False, save_data=False):
    print "beta = %2.2f, frames = %d" % (beta, frames)
    distance_list = []
    path_length = []
    for s in tqdm(range(num_of_strings)):
        d, pl = get_path_length_and_distances(beta, num_of_strings, L, frames,
                                              num_of_pairs)
        distance_list.append(d)
        path_length.append(pl)

    distance_list = np.array(distance_list).flatten()
    path_length = np.array(path_length).flatten()

    if save_data:
        # sd.save("results/data/distances/beta=%2.2f_" % beta,
        #                beta=beta, num_of_strings=num_of_strings,
        #                L=L, frames=frames, distance_list=distance_list,
        #                path_length=path_length)
        sd.save("results/data/distances/frames=%d_beta=%2.2f_" % (frames, beta),
                       beta=beta, num_of_strings=num_of_strings,
                       L=L, frames=frames, distance_list=distance_list,
                       path_length=path_length)

    if plot or save_image:
        fig, ax = plt.subplots()

        # heatmap
        ax.hist2d(distance_list, path_length, bins=25)

        ax.set_xlabel('Distance')
        ax.set_ylabel('Path length')
        ax.set_title('Path length and distances between two points in the cluster'
                    + r'($\beta = %2.2f$)' % beta)

        if save_image:
            result_image_path = "results/img/distances/beta=%2.2f" % beta
            result_image_path += "_" + time.strftime("%y%m%d_%H%M%S")
            result_image_path += ".png"
            plt.savefig(result_image_path)
            plt.close()
            print "[saved] " + result_image_path
        else:
            plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--beta', type=float, nargs=1,
                        help='parameter beta (inverse temparature)')
    parser.add_argument('--frames', type=int, nargs=1,
                        help='simulation frames')
    args = parser.parse_args()
    beta = args.beta[0]
    frames = args.frames[0]

    params = {
        'num_of_strings': 30,
        'L': (frames + 1) * 2,
        'frames': frames,
        'plot': False,
        'save_image': False,
        # 'num_of_pairs': 300,
        'num_of_pairs': 100,
        'save_data': True,
    }

    execute_simulation_for_one_beta(beta, **params)

