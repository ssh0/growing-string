#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-05-30

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from triangular import LatticeTriangular as LT
from base import Main as base
import numpy as np
import random
from tqdm import tqdm
from multiprocessing import Pool

def print_debug(arg):
    # print arg
    pass


class Main(base):

    def __init__(self, Lx=40, Ly=40, N=1, size=[10], plot=False):
        # Create triangular lattice with given parameters
        self.lattice = LT(np.zeros((Lx, Ly), dtype=np.int),
                          scale=10., boundary={'h': 'periodic', 'v': 'periodic'})

        self.occupied = np.zeros((Lx, Ly), dtype=np.bool)
        self.number_of_lines = sum(size)

        # Put the strings to the lattice
        self.strings = self.create_random_strings(1, size)

        # Record the number of time-steps to reach the deadlocks
        self.num_deadlock = 0

        self.plot = plot
        while True:
            try:
                self.update()
            except StopIteration:
                break

    def update(self, num=0):
        # numを記録して，ロックが起こるまでの時間を測る。
        # いくつかのstringサイズ，格子サイズの上でどのように変動するだろうか
        # 詳しいパラメータの設定などはメモ参照。
        # move head part of string (if possible)
        X = self.get_next_xy(self.strings[0].x, self.strings[0].y)
        if not X:
            print_debug(self.num_deadlock)
            raise StopIteration

        # update starting position
        x, y, vec = X
        rmx, rmy = self.strings[0].follow((x, y, (vec+3)%6))
        self.occupied[x, y] = True
        self.occupied[rmx, rmy] = False

        # Record time steps
        self.num_deadlock += 1

        # print self.occupied
        # print self.strings[0].pos, self.strings[0].vec

        if self.plot:
            ret = self.plot_string()
            return ret

    def get_next_xy(self, x, y):
        nnx, nny = self.lattice.neighbor_of8x, y)
        vectors = [i for i in range(6) if not self.occupied[nnx[i], nny[i]]]
        if len(vectors) == 0:
            print_debug("no neighbors")
            return False

        # 確率的に方向を決定
        vector = random.choice(vectors)
        # 点の格子座標を返す
        x, y = nnx[vector], nny[vector]
        return x, y, vector


# trial = 100
# trial = 3000 # for npy
trial = 10000
params = dict(Lx=40, Ly=40, plot=False)
# def calc_for_each_size(size):
#     summation = 0.
#     for t in range(trial):
#         main = Main(size=size, **params)
#         summation += main.num_deadlock
#     return summation / trial

def calc_for_each_size(size):
    ret = []
    for t in range(trial):
        main = Main(size=[size], **params)
        ret.append(main.num_deadlock)
    return ret


if __name__ == '__main__':
    # Simple observation of moving string.
    # main = Main(Lx=40, Ly=40, lattice_scale=10., size=100, plot=True)

    # Calcurate the deadlock time without plots.
    # main = Main(Lx=40, Ly=40, lattice_scale=10., size=10, plot=False)
    # print main.num_deadlock

    #==========================================================================
    # Create data
    pool = Pool(6)
    sizeset = np.unique(np.logspace(3., 8., num=50, base=2, dtype=np.int))
    it = pool.imap(calc_for_each_size, sizeset)
    T = []
    for ret in tqdm(it, total=len(sizeset)):
        T.append(ret)
    T = np.array(T)
    #==========================================================================

    #=-========================================================================
    # save the data for plotting, and so on
    # np.savez("2016-05-31.npz", trial=trial, sizeset=sizeset, T=T)
    # np.savez("2016-06-02.npz", trial=trial, sizeset=sizeset, T=T)
    # np.savez("2016-06-03_80.npz", trial=trial, sizeset=sizeset, T=T)
    # np.savez("2016-06-03_120.npz", trial=trial, sizeset=sizeset, T=T)
    # np.savez("2016-06-07_40.npz", trial=trial, sizeset=sizeset, T=T)
    # np.savez("2016-07-12_40.npz", trial=trial, sizeset=sizeset, T=T)
    #==========================================================================

    # プロット準備
    # fig, ax = plt.subplots()
    # ax.set_title("Deadlock time for the string size N on triangular lattice")

    #=0========================================================================
    # 普通に表示
    # ax.plot(sizeset, T, marker='o')
    # ax.set_xlabel("$N$")
    # ax.set_ylabel("$T$")
    # 反比例のように見えた
    #==========================================================================

    #=1========================================================================
    # loglogで表示
    # ax.loglog(sizeset, T, marker='o')
    # ax.set_xlabel("$N$")
    # ax.set_ylabel("$T$")
    # 反比例のように見えた
    #==========================================================================

    #=2========================================================================
    # 1/log(N)とlog(T)の関係を見た
    # logsizeset = np.log10(sizeset)
    # logT = np.log10(T)
    # ax.plot(1 / logsizeset, logT, marker='o')
    # ax.set_xlabel("$N$")
    # ax.set_ylabel("$T$")
    # 厳密には直線ではなさそうだった。
    #==========================================================================

    # plt.show()
