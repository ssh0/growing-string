#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-08-17

from growing_string import Main
import matplotlib.pyplot as plt
import numpy as np


def is_equal(a, b):
    if a == b:
        return 1
    else:
        return 0

is_equal = np.vectorize(is_equal)

def get_straight_lines(l):
    L0, L1 = [], []
    x = l[0]
    if x == 0:
        l0, l1 = 1, 0
    else:
        l0, l1 = 0, 1

    for p in l[1:]:
        if p == 0:
            if x == 0:
                l0 += 1
            else:
                l0 = 1
                L1.append(l1)
                l1 = 0
        else:
            if x == 0:
                l1 = 1
                L0.append(l0)
                l0 = 0
            else:
                l1 += 1
        x = p

    if l0 == 0:
        L1.append(l1)
    else:
        L0.append(l0)
    return L1

def is_straight(self, i, s):
    return get_straight_lines(is_equal(s.vec[:-1], s.vec[1:]))

if __name__ == '__main__':
    main = Main(Lx=50, Ly=50, size=[3,] * 1, plot=True, frames=100,
                pre_function=is_straight)
    l = np.array(main.pre_func_res).T
    print l
    # steps = range(len(l))
    # fig, ax = plt.subplots()
    # ax.set_xlabel("Step")
    # ax.set_ylabel("Number of pairs")
    # ax.legend(loc="best")
    # plt.show()
