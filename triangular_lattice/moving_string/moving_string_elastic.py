#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-06-06

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from triangular import LatticeTriangular as LT
from string import String
from base import Main as base
import numpy as np
import random


def print_debug(arg):
    # print arg
    pass


class Main(base):

    def __init__(self, Lx=40, Ly=40, N=4, size=[5, 4, 10, 12], plot=True):
        # Create triangular lattice with given parameters
        self.lattice = LT(np.zeros((Lx, Ly), dtype=np.int),
                          scale=10., boundary={'h': 'periodic', 'v': 'periodic'})

        self.occupied = np.zeros((Lx, Ly), dtype=np.bool)
        self.number_of_lines = sum(size)

        # Put the strings to the lattice
        self.strings = self.create_random_strings(N, size)

        self.plot = plot
        self.interval = 100

    def update(self, num=0):
        # move head part of each strings (if possible)
        for s in self.strings:
            X = self.get_next_xy(s.x, s.y, s.vec[0])
            if not X:
                raise StopIteration

            # update starting position
            x, y, vec = X
            rmx, rmy = s.follow((x, y, (vec + 3) % 6))
            self.occupied[x, y] = True
            self.occupied[rmx, rmy] = False

        ret = self.plot_string()

        if self.plot:
            ret = self.plot_string()
            return ret

    def get_next_xy(self, x, y, vec):
        # 曲げ弾性の効果を再現するために，位置関係によって次の点の
        # 選ばれやすさが異なるようにする
        nnx, nny = self.lattice.neighbor_of(x, y)
        vectors = [i for i in range(6) if not self.occupied[nnx[i], nny[i]]]
        if len(vectors) == 0:
            print_debug("no neighbors")
            return False

        # 確率的に方向を決定
        # 先頭ベクトルを0とした時の相対ベクトルごとの選ばれやすさを設定
        # weights = [0., 1., 2., 3., 2., 1.]
        weights = [0., 0., 1., 4., 1., 0.]
        # weights = [0., 0., 2., 1., 2., 0.]
        # weights = [0., 0., 0., 1., 0., 0.]
        # 有効なものだけ取り出す
        weights = np.array([weights[(i + 6 - vec) % 6] for i in vectors])
        # 規格化
        weights = weights / np.sum(weights)
        # vectorsから一つ選択
        vector = np.random.choice(vectors, p=weights)
        # 点の格子座標を返す
        x, y = nnx[vector], nny[vector]
        return x, y, vector

    def create_random_strings(self, N=3, size=[10, 5, 3]):
        """Create N strings of each size is specified with 'size'.

        This process is equivalent to self-avoiding walk on triangular lattice.
        """
        strings = []

        n = 0
        while n < N:
            # set starting point
            x = random.randint(0, self.lattice.Lx - 1)
            y = random.randint(0, self.lattice.Ly - 1)
            if self.occupied[x, y]:  # reset
                print_debug("(%d, %d) is occupied already. continue." % (x, y))
                continue
            self.occupied[x, y] = True

            S = size[n]
            pos = [(x, y, np.random.choice(6))]
            trial, nmax = 0, 10
            double = 0
            while len(pos) < S:
                if trial > nmax:
                    for _x, _y, vec in pos:
                        self.occupied[_x, _y] = False
                    print_debug("All reset")
                    break
                X = self.get_next_xy(x, y, pos[-1][2])
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
                strings.append(String(self.lattice, n, pos[0][0], pos[0][1],
                                      vec=vec))
                n += 1

        return strings


if __name__ == '__main__':
    N = 10
    main = Main(Lx=40, Ly=40, N=N, size=[random.randint(4, 8) for i in range(N)])
    # Plot triangular-lattice points, string on it, and so on
    if main.plot:
        main.plot_all()
    else:
        while True:
            try:
                main.update()
            except StopIteration:
                break
