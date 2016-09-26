#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-09-20
"""Eden model on triangular lattice"""


from triangular import LatticeTriangular as LT
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.animation as animation
import numpy as np
import random


def print_debug(arg):
    """Print argument if needed.

    You can use this function in any parts and its behavior is toggled here.
    """
    # print arg
    pass


class Eden():

    def __init__(self, Lx=60, Ly=60, plot=True, frames=1000,
                 boundary='periodic',
                 pre_function=None,
                 post_function=None):
        # Create triangular lattice with given parameters
        self.lattice = LT(np.zeros((Lx, Ly), dtype=np.int),
                          scale=float(max(Lx, Ly)), boundary=boundary)

        self.lattice_X = self.lattice.coordinates_x
        self.lattice_Y = self.lattice.coordinates_y
        self.lattice_X = self.lattice_X.reshape(self.lattice.Lx,
                                                self.lattice.Ly)
        self.lattice_Y = self.lattice_Y.reshape(self.lattice.Lx,
                                                self.lattice.Ly)

        self.occupied = np.zeros((Lx, Ly), dtype=np.bool)

        center_x, center_y = Lx / 2, Ly / 4
        self.points = [(center_x, center_y)]
        self.occupied[center_x, center_y] = True
        self.neighbors = list(map(
            tuple, self.lattice.neighborhoods[center_x, center_y].T
        ))

        self.plot = plot
        self.interval = 1
        self.frames = frames

        self.pre_function = pre_function
        self.post_function = post_function
        self.pre_func_res = []
        self.post_func_res = []

    def execute(self):
        if self.plot:
            self.plot_all()
        else:
            t = 0
            while t < self.frames:
                try:
                    self.update()
                    t += 1
                except StopIteration:
                    break

    def plot_all(self):
        """軸の設定，三角格子の描画

        ここからFuncAnimationを使ってアニメーション表示を行うようにする
        """

        self.fig, self.ax = plt.subplots(figsize=(8, 8))

        self.lattice_X = self.lattice.coordinates_x
        self.lattice_Y = self.lattice.coordinates_y
        X_min, X_max = min(self.lattice_X) - 0.1, max(self.lattice_X) + 0.1
        Y_min, Y_max = min(self.lattice_Y) - 0.1, max(self.lattice_Y) + 0.1
        self.ax.set_xlim([X_min, X_max])
        self.ax.set_ylim([Y_min, Y_max])
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.set_aspect('equal')

        triang = tri.Triangulation(self.lattice_X, self.lattice_Y)
        self.ax.triplot(triang, color='#d5d5d5', marker='.', markersize=1)

        self.scatter = [
            self.ax.plot([], [], '.', color='black')[0],
            self.ax.plot([], [], 'o', color='#d5d5d5')[0]
        ]
        plt.scatter

        self.lattice_X = self.lattice_X.reshape(self.lattice.Lx,
                                                self.lattice.Ly)
        self.lattice_Y = self.lattice_Y.reshape(self.lattice.Lx,
                                                self.lattice.Ly)
        self.plot_points()

        def init_func(*arg):
            return self.scatter

        ani = animation.FuncAnimation(self.fig, self.update, frames=self.frames,
                                      init_func=init_func,
                                      interval=self.interval,
                                      blit=True, repeat=False)
        plt.show()

    def plot_points(self):
        """self.pointsに格納されている格子座標を元にプロット
        """
        points = self.points  # [(x1, y1), (x2, y2), (x3, y3), ...]
        index = list(np.array(points).T)
        X = self.lattice_X[index]
        Y = self.lattice_Y[index]
        self.scatter[0].set_data(X, Y)

        index = list(np.array(self.neighbors).T)
        nx = self.lattice_X[index]
        ny = self.lattice_Y[index]
        self.scatter[1].set_data(nx, ny)
        return self.scatter

    def update(self, num=0):
        """funcanimationから各フレームごとに呼び出される関数

        1時間ステップの間に行う計算はすべてここに含まれる。
        """

        if len(self.neighbors) == 0:
            print_debug("no neighbors")
            return False

        if self.pre_function is not None:
            self.pre_func_res.append(self.pre_function(self))

        x, y = self.neighbors.pop(random.randint(0, len(self.neighbors) - 1))
        self.occupied[x, y] = True
        self.points.append((x, y))

        new_n = set(((nx, ny) for nx, ny in self.lattice.neighborhoods[x, y].T
                    if nx != -1 and ny != -1))
        updated_n = set(tuple(map(tuple, self.neighbors))) | new_n
        self.neighbors = [pos for pos in updated_n if not self.occupied[pos]]

        if self.post_function is not None:
            self.post_func_res.append(self.post_function(self))

        if self.plot:
            return self.plot_points() 


if __name__ == '__main__':
    eden = Eden()
    # eden = Eden(plot=False)
    eden.execute()

    print_debug(eden.occupied)
    print_debug(len(eden.neighbors))
    print_debug(len(np.where(eden.occupied)[0]))
    print_debug(len(eden.points))
    print_debug(eden.points)

