#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2017-03-22
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from triangular import LatticeTriangular as LT
from triangular_random import randomize
from random_lattice.base_random_lattice import Main as base
from strings import String
import matplotlib.tri as tri
import numpy as np
import random


class Main(base):

    def __init__(self, Lx=40, Ly=40, N=4, size=[5, 4, 10, 12], plot=True,
                 beta=4.):
        self.plot = plot
        self.beta = beta
        self.interval = 50
        self.frames = 1000

        # Create triangular lattice with given parameters
        self.lattice = LT(np.zeros((Lx, Ly), dtype=np.int),
                          scale=10.,
                          boundary={'h': 'periodic', 'v': 'periodic'}
                          )
        randomize(self.lattice)
        self.triang_standard = tri.Triangulation(self.lattice.coordinates_x,
                                                 self.lattice.coordinates_y)
        self.triang_random = tri.Triangulation(
            self.lattice.coordinates_x,
            self.lattice.coordinates_y,
            triangles=self.triang_standard.triangles)

        self.lattice_X = self.lattice.coordinates_x.reshape(
            self.lattice.Lx,
            self.lattice.Ly
        )
        self.lattice_Y = self.lattice.coordinates_y.reshape(
            self.lattice.Lx,
            self.lattice.Ly
        )
        self.occupied = np.zeros((Lx, Ly), dtype=np.bool)
        self.number_of_lines = sum(size)

        # Put the strings to the lattice
        self.strings = self.create_random_strings(N, size)

    def _vector(self, X1, Y1, X2, Y2):
        """Return 2-d vector from (X1, Y1) to (X2, Y2).
        Remark: X1, X2,... are grid coordinates and are not coordinates in
        real 2d space."""
        x1, y1 = self.lattice_X[X1][Y1], self.lattice_Y[X1][Y1]
        x2, y2 = self.lattice_X[X2][Y2], self.lattice_Y[X2][Y2]
        return np.array([x2 - x1, y2 - y1])

    def _vec_(self, s, i):
        """Get vector from the point in string 's' indexed 'i' to the point
        indexed 'i+1'."""
        X1, Y1 = s.pos[i]
        X2, Y2 = s.pos[i + 1]
        return self._vector(X1, Y1, X2, Y2)

    def dot(self, vec1, vec2):
        """Calculate user-defined 'inner product' from the two 2d vectors
        `vec1` and `vec2`.
        """
        ## normal inner product
        # res = np.dot(vec1, vec2)

        ## only depend on angle
        res = np.dot(vec1 / np.linalg.norm(vec1), vec2 / np.linalg.norm(vec2))
        return res

    def update(self, num=0):
        # move head part of each strings (if possible)
        for s in self.strings:
            X = self.get_next_xy(s.x, s.y, self._vec_(s, 0))
            if not X:
                ## dead lockの縛りはなくす
                # raise StopIteration
                continue

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
            # print_debug("no neighbors")
            return False

        # 確率的に方向を決定
        if type(vec) == np.ndarray:
            weights = np.array([
                np.exp(
                    self.beta * self.dot(self._vector(nnx[i], nny[i], x, y), vec)
                ) for i in vectors])
        else:
            weights = np.array([1.] * len(vectors))
        # 規格化
        weights = weights / np.sum(weights)
        # vectorsから一つ選択
        selected_vector = np.random.choice(vectors, p=weights)
        # 点の格子座標を返す
        x, y = nnx[selected_vector], nny[selected_vector]
        return x, y, selected_vector

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
                # print_debug("(%d, %d) is occupied already. continue." % (x, y))
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
                    # print_debug("All reset")
                    break
                if len(pos) == 1:
                    X = self.get_next_xy(x, y, 1)
                else:
                    X = self.get_next_xy(x, y, self._vector(
                        pos[-2][0], pos[-2][1], pos[-1][0], pos[-1][1]))
                if not X:
                    if len(pos) == 1:
                        # print_debug("len(pos) == 1")
                        double = 0
                        trial += 1
                        break
                    else:
                        # print_debug("back one step")
                        # print_debug(pos)
                        if double == 1:
                            # print_debug("two time. back two step")
                            # print_debug(pos)
                            oldx, oldy, oldvec = pos[-1]
                            del pos[-1]
                            self.occupied[oldx, oldy] = False
                            oldx, oldy, oldvec = pos[-1]
                            del pos[-1]
                            self.occupied[oldx, oldy] = False
                            x, y, vector = pos[-1]
                            trial += 1
                            # print_debug(pos)
                            break
                        oldx, oldy, oldvec = pos[-1]
                        del pos[-1]
                        self.occupied[oldx, oldy] = False
                        x, y, vector = pos[-1]
                        trial += 1
                        # print_debug(pos)
                        continue
                else:
                    double = 0
                    x, y, vector = X
                    self.occupied[x, y] = True
                    pos.append((x, y, vector))
                    # print_debug("add step normally")
                    # print_debug(pos)
            else:
                # print_debug("Done. Add string")
                vec = [v[2] for v in pos][1:]
                strings.append(String(self.lattice, n, pos[0][0], pos[0][1],
                                      vec=vec))
                n += 1

        return strings


if __name__ == '__main__':
    N = 30
    # main = Main(Lx=100, Ly=100, N=N, size=[random.randint(4, 12)] * N)
    main = Main(Lx=100, Ly=100, N=N, size=[random.randint(8, 20)] * N, beta=20.)

    # Plot triangular-lattice points, string on it, and so on
    if main.plot:
        main.plot_all()
    else:
        while True:
            try:
                main.update()
            except StopIteration:
                break
