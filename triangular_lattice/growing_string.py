#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-05-22

from triangular import LatticeTriangular as LT
from String import String
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.animation as animation
import numpy as np
from numpy import linalg as la
import random
import time

def print_debug(arg):
    # print arg
    pass


class Main:

    def __init__(self, Lx=40, Ly=40, lattice_scale=10., N=4, size=[5, 4, 10, 12]):
        # Create triangular lattice with given parameters
        self.lattice = LT(np.zeros((Lx, Ly), dtype=np.int),
                          scale=lattice_scale, boundary='periodic')

        self.occupied = np.zeros((Lx, Ly), dtype=np.bool)
        self.number_of_lines = sum(size)

        # Put the strings to the lattice
        self.strings = self.create_random_strings(N, size)

        # Plot triangular-lattice points, string on it, and so on
        self.plot_all()

    def plot_all(self):
        frames = 1000
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
                                      interval=50, blit=True, repeat=False)
        plt.show()

    def plot_string(self):
        # print self.string.pos, self.string.vec

        i = 0  # to count how many line2D object
        for s in self.strings:
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
        # move head part of each strings (if possible)
        for s in self.strings:
            X = self.get_next_xy(s.x, s.y)
            if not X:
                raise StopIteration

            # update starting position
            x, y, vec = X
            rmx, rmy = s.follow((x, y, (vec+3)%6))
            self.occupied[x, y] = True
            self.occupied[rmx, rmy] = False

        ret = self.plot_string()

        # print self.occupied
        # for s in self.strings:
        #     print s.pos, s.vec

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
                strings.append(String(self.lattice, n, pos[0][0], pos[0][1],
                                      vec=vec))
                n += 1

        return strings

if __name__ == '__main__':
    # main = Main()
    main = Main(N=10, size=[random.randint(4, 12)]*10)
