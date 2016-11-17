#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-10-07
"""頂点間距離とベクトルの向きの相関
"""

from growing_string import Main
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm
import time
import save_data


def choose_indexes(_list, num, L):
    """Choose the index pairs whose width is fixed. """
    N = len(_list)
    if N - (2 * L) < num:
        raise StopIteration('list index is smaller than expected (%d), '
                            % (num + 2 * L)
                            + 'given (%d).' % N
                            )
    return sorted(random.sample(_list[L:N - L], num))

def _to_radian(i, j):
    k = (i + 6 - j) % 6
    if k == 0: return 0.
    if k == 1: return np.pi / 3
    if k == 2: return 2 * np.pi / 3
    if k == 3: return np.pi
    if k == 4: return 4 * np.pi / 3
    if k == 5: return 5 * np.pi / 3

to_radian = np.vectorize(_to_radian)

def calc_order_param(theta):
    itheta = np.array([1j*t for t in theta])
    R = abs(np.sum(np.exp(itheta))/float(len(theta)))
    return R

def get_correlation(beta, num_of_strings, L, frames, num_of_pairs=300):
    len_vec = frames + 2
    Lp = range(2, (len_vec - num_of_pairs) / 2)
    _Cs = []
    for s in tqdm(range(num_of_strings)):
        _Cs.append(get_correlation_for_each_string(Lp, L, frames, num_of_pairs))
    Cs = np.average(np.array(_Cs), axis=0)
    return Lp, Cs

def get_correlation_for_each_string(Lp, L, frames, num_of_pairs):
    main = Main(Lx=L, Ly=L, plot=False, frames=frames, beta=beta,
                strings=[{'id': 1, 'x': L/4, 'y': L/2, 'vec': [0, 4]}])
    len_vec = len(main.strings[0].vec)

    # 等パス長となる2点を同数ずつ抽出
    random_i, random_j = [], []
    for lp in Lp:
        random_i.append(np.array(choose_indexes(range(len_vec),
                                                num_of_pairs, lp)))
        random_j.append(random_i[-1] + lp)

    random_i = np.array(random_i).flatten()
    random_j = np.array(random_j).flatten()

    # 2. 各点でのベクトルの相関を計算
    vec0 = np.array(main.strings[0].vec)[random_i].reshape((len(Lp), num_of_pairs))
    vec1 = np.array(main.strings[0].vec)[random_j].reshape((len(Lp), num_of_pairs))
    # ペア間の角度
    rad = to_radian(vec0, vec1)
    # 角度の揃い具合を計算
    Cs = [calc_order_param(rad[i]) for i in range(len(Lp))]

    return Cs


if __name__ == '__main__':

    start_time = time.strftime("%y%m%d_%H%M%S")
    num_of_strings = 30
    betas = [0., 5., 10., 15., 20.]
    # betas = [float(i) for i in range(11)]
    # betas = [20.]
    frames = 1000
    L = 1000
    num_of_pairs = 300

    fig, ax = plt.subplots()

    for beta in betas:
        print "beta = %2.2f" % beta
        Lp, Cs = get_correlation(beta, num_of_strings, L, frames, num_of_pairs)
        ax.plot(Lp, Cs, '.', label=r'$\beta = %2.2f$' % beta)

        # save the data
        save_data.save("results/data/correlation/beta=%2.2f_" % beta,
                       num_of_strings=num_of_strings,
                       beta=beta, L=L, frames=frames, Lp=Lp, Cs=Cs)

    ax.set_xlabel('Path length')
    ax.set_ylabel('Correlation of the vectors')
    ax.set_title('Correlation of the vectors')
    ax.legend(loc='best')

    result_image_path = "results/img/correlation/strings=%d" % num_of_strings
    result_image_path += "_" + start_time
    result_image_path += ".png"
    plt.savefig(result_image_path)
    plt.close()
    print "[saved] " + result_image_path

    # plt.show()
