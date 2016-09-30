#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-07-29
"""ステップ数とクラスター内の二つのベクトルペアの折れ曲がりの3状態をカウント
"""

from growing_string import Main
import matplotlib.pyplot as plt
import numpy as np
from optimize import Optimize_powerlaw


def is_equal(a, b):
    if a == b:
        return 1
    else:
        return 0

def is_bend1(a, b):
    ang = (a + 6 - b) % 6
    if ang in (1, 5):
        return 1
    else:
        return 0

def is_bend2(a, b):
    ang = (a + 6 - b) % 6
    if ang in (2, 4):
        return 1
    else:
        return 0

is_equal = np.vectorize(is_equal)
is_bend1 = np.vectorize(is_bend1)
is_bend2 = np.vectorize(is_bend2)

def count_bend(self, i, s):
    pairs = np.sum(is_equal(s.vec[:-1], s.vec[1:]))
    bend1 = np.sum(is_bend1(s.vec[:-1], s.vec[1:]))
    bend2 = np.sum(is_bend2(s.vec[:-1], s.vec[1:]))
    return [pairs, bend1, bend2]

if __name__ == '__main__':
    main = Main(Lx=60, Ly=60, size=[3,] * 1, plot=False, frames=1000,
                beta=0.,
                pre_function=count_bend)
    pairs, bend1, bend2 = np.array(main.pre_func_res).T
    steps = range(len(pairs))

    # fig, ax = plt.subplots()
    # ax.plot(steps, pairs, label="straight")
    # ax.plot(steps, bend1, label="bend1")
    # ax.plot(steps, bend2, label="bend2")
    # ax.set_xlabel("Step")
    # ax.set_ylabel("Number of pairs")
    # ax.legend(loc="best")
    # plt.show()

    # pairsがべき乗則に従いそう。
    # フィッティングを行う。
    optimizer = Optimize_powerlaw(args=(steps, pairs), parameters=[0., 1.])
    result = optimizer.fitting()
    print "D = %f" % result['D']
    # # D ≒ 1.0

    fig, ax = plt.subplots()
    ax.plot(steps, pairs, label="straight")
    ax.plot(steps, optimizer.fitted(steps), label='D = %f' % result['D'])
    ax.set_xlabel("Step")
    ax.set_ylabel("Number of pairs")
    ax.legend(loc="best")
    plt.show()
