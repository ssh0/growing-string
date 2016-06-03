#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-05-30

from triangular import LatticeTriangular as LT
from String import String
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.animation as animation
import numpy as np
from numpy import linalg as la
import random
import time
import sys
from tqdm import tqdm
from multiprocessing import Pool

def print_debug(arg):
    # print arg
    pass


class Main:

    def __init__(self, Lx=40, Ly=40, lattice_scale=10., size=10, plot=True):
        # Create triangular lattice with given parameters
        self.lattice = LT(np.zeros((Lx, Ly), dtype=np.int),
                          scale=lattice_scale, boundary='periodic')

        self.occupied = np.zeros((Lx, Ly), dtype=np.bool)
        self.number_of_lines = size

        # Put the string to the lattice
        self.string = self.create_random_string(size)

        # Record the number of time-steps to reach the deadlocks
        self.num_deadlock = 0

        self.plot = plot

        # Plot triangular-lattice points, string on it, and so on
        if self.plot:
            self.plot_all()
        else:
            while True:
                try:
                    self.update()
                except StopIteration:
                    break

    def plot_all(self):
        frames = 10000
        self.fig, self.ax = plt.subplots(figsize=(8, 8))

        self.lattice_X = self.lattice.coordinates_x
        self.lattice_Y = self.lattice.coordinates_y

        X_min, X_max = min(self.lattice_X) - 0.1, max(self.lattice_X) + 0.1
        Y_min, Y_max = min(self.lattice_Y) - 0.1, max(self.lattice_Y) + 0.1
        self.ax.set_xlim([X_min, X_max])
        self.ax.set_ylim([Y_min, Y_max])

        triang = tri.Triangulation(self.lattice_X, self.lattice_Y)
        self.ax.triplot(triang, color='#d5d5d5', marker='.', markersize=1)

        self.lines = [self.ax.plot([], [], marker='o', linestyle='-',
                                   color='black',
                                   markerfacecolor='black',
                                   markeredgecolor='black')[0]
                      for i in range(self.number_of_lines)]

        self.lattice_X = self.lattice_X.reshape(self.lattice.Lx,
                                                self.lattice.Ly)
        self.lattice_Y = self.lattice_Y.reshape(self.lattice.Lx,
                                                self.lattice.Ly)
        self.plot_string()

        ani = animation.FuncAnimation(self.fig, self.update, frames=frames,
                                      interval=1, blit=True, repeat=False)
        plt.show()

    def plot_string(self):
        # print self.string.pos, self.string.vec

        i = 0  # to count how many line2D object
        s = self.string
        start = 0
        for j, pos1, pos2 in zip(range(len(s.pos) - 1), s.pos[:-1], s.pos[1:]):
            dist_x = abs(self.lattice_X[pos1[0], pos1[1]] - self.lattice_X[pos2[0], pos2[1]] )
            dist_y = abs(self.lattice_Y[pos1[0], pos1[1]] - self.lattice_Y[pos2[0], pos2[1]] )
            # print j, pos1, pos2
            # print dist_x, dist_y
            if dist_x > 1.5 * self.lattice.dx or dist_y > 1.5 * self.lattice.dy:
                x = s.pos_x[start:j+1]
                y = s.pos_y[start:j+1]
                X = [self.lattice_X[_x, _y] for _x, _y in zip(x, y)]
                Y = [self.lattice_Y[_x, _y] for _x, _y in zip(x, y)]
                self.lines[i].set_data(X, Y)
                start = j + 1
                i += 1
        else:
            x = s.pos_x[start:]
            y = s.pos_y[start:]
            X = [self.lattice_X[_x, _y] for _x, _y in zip(x, y)]
            Y = [self.lattice_Y[_x, _y] for _x, _y in zip(x, y)]
            self.lines[i].set_data(X, Y)
            i += 1
        # 最終的に，iの数だけ線を引けばよくなる
        # それ以上のオブジェクトはリセット
        for j in range(i, len(self.lines)):
            self.lines[j].set_data([], [])

        return self.lines

    def update(self, num=0):
        # TODO: numを記録して，ロックが起こるまでの時間を測る。
        # いくつかのstringサイズ，格子サイズの上でどのように変動するだろうか
        # 詳しいパラメータの設定などはメモ参照。
        # move head part of string (if possible)
        X = self.get_next_xy(self.string.x, self.string.y)
        if not X:
            print_debug(self.num_deadlock)
            raise StopIteration

        # update starting position
        x, y, vec = X
        rmx, rmy = self.string.follow((x, y, (vec+3)%6))
        self.occupied[x, y] = True
        self.occupied[rmx, rmy] = False

        # Record time steps
        self.num_deadlock += 1

        # print self.occupied
        # print self.string.pos, self.string.vec

        if self.plot:
            ret = self.plot_string()
            return ret

    def get_next_xy(self, x, y):
        nnx, nny = self.lattice.neighborhoods[x, y]
        vectors = [i for i in range(6) if not self.occupied[nnx[i], nny[i]]]
        if len(vectors) == 0:
            print_debug("no neighbors")
            return False

        # 確率的に方向を決定
        vector = random.choice(vectors)
        # 点の格子座標を返す
        x, y = nnx[vector], nny[vector]
        return x, y, vector

    def create_random_string(self, size=10):
        """Create a string which size is specified with 'size'.

        This process is equivalent to self-avoiding walk on triangular lattice.
        """

        n, N = 0, 1
        while n < N:
            # set starting point
            x = random.randint(0, self.lattice.Lx - 1)
            y = random.randint(0, self.lattice.Ly - 1)
            if self.occupied[x, y]:  # reset
                print_debug("(%d, %d) is occupied already. continue." % (x, y))
                continue
            self.occupied[x, y] = True

            S = size
            pos = [(x, y, 'dummy')]
            trial, nmax = 0, 10
            double = 0
            while len(pos) < S:
                if trial > nmax:
                    for _x, _y, vec in pos:
                        self.occupied[_x, _y] = False
                    print_debug("All reset")
                    break
                X = self.get_next_xy(x, y)
                if not X:
                    if len(pos) == 1:
                        print_debug("len(pos) == 1")
                        double = 0
                        trial += 1
                        break
                    else:
                        print_debug("back one step")
                        print_debug(pos)
                        if double == 1:
                            print_debug("two time. back two step")
                            print_debug(pos)
                            oldx, oldy, oldvec = pos[-1]
                            del pos[-1]
                            self.occupied[oldx, oldy] = False
                            oldx, oldy, oldvec = pos[-1]
                            del pos[-1]
                            self.occupied[oldx, oldy] = False
                            x, y, vector = pos[-1]
                            trial += 1
                            print_debug(pos)
                            break
                        oldx, oldy, oldvec = pos[-1]
                        del pos[-1]
                        self.occupied[oldx, oldy] = False
                        x, y, vector = pos[-1]
                        trial += 1
                        print_debug(pos)
                        continue
                else:
                    double = 0
                    x, y, vector = X
                    self.occupied[x, y] = True
                    pos.append((x, y, vector))
                    print_debug("add step normally")
                    print_debug(pos)
            else:
                print_debug("Done. Add string")
                vec = [v[2] for v in pos][1:]
                string = String(self.lattice, n, pos[0][0], pos[0][1], vec=vec)
                n += 1

        return string


trial = 3000 # for npy
# trial = 1000
params = dict(Lx=40, Ly=40, lattice_scale=10, plot=False)
# def calc_for_each_size(size):
#     summation = 0.
#     for t in range(trial):
#         main = Main(size=size, **params)
#         summation += main.num_deadlock
#     return summation / trial

def calc_for_each_size(size):
    ret = []
    for t in range(trial):
        main = Main(size=size, **params)
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
    pool = Pool(processes=6)
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
    np.savez("2016-06-02.npz", trial=trial, sizeset=sizeset, T=T)
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
