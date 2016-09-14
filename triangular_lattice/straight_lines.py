#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-08-17

from growing_string import Main
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

# constants
num_strings = 10
steps = 1000
L = 60
# note: stepsとLの大きさに注意
# (stepsが大きすぎると無限大格子状で成長させるのとは異なる結果となる)

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

def is_straight(_self, i, s):
    return get_straight_lines(is_equal(s.vec[:-1], s.vec[1:]))

def count_straight_lines(s):
    # main = Main(Lx=L, Ly=L, size=[3,] * 1, plot=False, frames=steps,
    #             pre_function=is_straight)
    main = Main(Lx=L, Ly=L, size=[3,] * 1, plot=False, frames=steps,
                dot_alpha=2.,
                pre_function=is_straight)
    return main.pre_func_res

def pad_list(lst):
    n = max(map(len, lst))
    res = np.zeros([len(lst), n])
    for i, row in enumerate(lst):
        for j, val in enumerate(row):
            res[i][j] = val
    return res

if __name__ == '__main__':

    pool = Pool(7)

    straight_lines = []
    ite = pool.imap(count_straight_lines, range(num_strings))
    for ret in tqdm(ite, total=num_strings):
        straight_lines.append(ret)
    # print straight_lines

    hist_list = []
    for n in range(steps):
        _straight_lines = []
        for s in range(num_strings):
            _straight_lines += straight_lines[s][n]
        X = range(max(_straight_lines) + 1 if len(_straight_lines) > 0 else 2)
        hist = np.histogram(_straight_lines, bins=[x + 0.5 for x in X])[0]
        hist_list.append(hist)

    histogram = pad_list(hist_list) / float(num_strings)
    # print histogram

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(range(1, len(histogram[0]) + 1), range(steps))
    ax.plot_wireframe(X, Y, np.log(histogram), rstride=steps / 20)
    ax.set_title("Series of $n$ straight lines per one string cluster")
    ax.set_ylim(0, steps)
    ax.set_xlabel("Series of $n$ straight lines")
    ax.set_ylabel("Steps")
    ax.set_zlabel("Frequency")
    plt.show()
