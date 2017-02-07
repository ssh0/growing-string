#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-05-15

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from triangular import LatticeTriangular as LT
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.animation as animation
import numpy as np
from numpy import linalg as la
import random
import time


rint = random.randint
randm = random.random


class Point:

    def __init__(self, id, ix, iy):
        self.id, self.x, self.y = id, ix, iy
        # vel is unified and the value of it implies the direction of the
        # velocity
        self.vel = rint(0, 5)
        self.priority = randm()


class Main:

    def __init__(self, Lx=20, Ly=20, rho=0.9, lattice_scale=10, T=0.4, plot=True,
                 frames=100):
        self.lattice = LT(- np.ones((Lx, Ly), dtype=np.int),
                          scale=lattice_scale)
        self.N = int(Lx * Ly * rho)
        self.points = [Point(n, rint(0, Lx - 1), rint(0, Ly - 1)) for n
                       in range(self.N)]
        self.T = T
        self.plot = plot
        self.beta = 1. / self.T
        self.order_param = []
        self.num = 0
        angs = [i * np.pi / 3. for i in range(6)]
        self.velx = [np.cos(ang) for ang in angs]
        self.vely = [-np.sin(ang) for ang in angs]
        self.u = [np.array([vx, -vy]) for vx, vy in zip(self.velx, self.vely)]

        self.lattice_X = self.lattice.coordinates_x
        self.lattice_Y = self.lattice.coordinates_y
        self.lattice_X = np.array(self.lattice_X).reshape(Lx, Ly)
        self.lattice_Y = np.array(self.lattice_Y).reshape(Lx, Ly)
        X_min, X_max = np.min(self.lattice_X), np.max(self.lattice_X)
        Y_min, Y_max = np.min(self.lattice_Y), np.max(self.lattice_Y)

        if self.plot:
            self.fig, (self.ax1, self.ax2) = plt.subplots(
                1, 2, figsize=(8, 10))
            self.ax1.set_xlim([X_min, X_max])
            self.ax1.set_ylim([Y_min, Y_max])
            self.ax1.set_xticklabels([])
            self.ax1.set_yticklabels([])
            self.ax1.set_aspect('equal')
            self.ax1.set_title("Lattice-Gas model for collective motion")
            self.triang = tri.Triangulation(self.lattice_X.flatten(),
                                            self.lattice_Y.flatten())
            self.ax1.triplot(self.triang, color='whitesmoke', lw=0.5)

            self.l, = self.ax2.plot([], [], 'b-')
            self.ax2.set_title(r"Order parameter $m=\frac{1}{N} |\sum \vec{u}_{i}|$ ($T = %.2f$)"
                               % self.T)
            self.ax2.set_ylim([0, 1])

            def init_func(*arg):
                return self.l,

            ani = animation.FuncAnimation(self.fig, self.update, frames=frames,
                                          init_func=init_func,
                                          interval=1, blit=True, repeat=False)
            plt.show()
        else:
            for i in range(100):
                self.update(i)
            print self.order_param[-1]

    def update(self, num):
        lowest, upper = {}, []
        # 同じサイトにいるものを検出
        for point in self.points:
            if not lowest.has_key((point.x, point.y)):
                lowest[(point.x, point.y)] = point
            elif lowest[(point.x, point.y)].priority > point.priority:
                upper.append(lowest[(point.x, point.y)])
                lowest[(point.x, point.y)] = point
            else:
                upper.append(point)

        # priority値最小のものだけ最近接効果(decided by Boltzmann eq)をうける
        for point in lowest.values():
            # 最近接の速度の合計を求める
            velocities = np.array([0., 0.])
            nnx, nny = self.lattice.neighbor_of(point.x, point.y)
            for x, y in zip(nnx, nny):
                if lowest.has_key((x, y)):
                    ang = lowest[(x, y)].vel
                    velocities += np.array([self.velx[ang], -self.vely[ang]])

            # ボルツマン分布に従って確率的に方向を決定
            A = [np.exp(self.beta * np.dot(u, velocities)) for u in self.u]
            rand = randm() * sum(A)
            p = 0
            for i, P in enumerate(A):
                p += P
                if rand < p:
                    point.vel = i
                    break

        # それ以外はランダムに向きを変えるように
        for point in upper:
            # change the velocity of the point
            point.vel = rint(0, 5)

        # 各点の座標とベクトルを更新し，描画
        self.update_quivers()

        # オーダーパラメーターをプロット
        self.plot_order_param(num)

        return self.quiver, self.l

    def update_quivers(self):
        # Get information to plot
        X, Y = [], []
        for point in self.points:
            # Get possible direction
            newx, newy = self.lattice.neighbor_of(point.x, point.y)
            # Choose one by its velocity
            point.x, point.y = newx[point.vel], newy[point.vel]
            X.append(self.lattice_X[point.x, point.y])
            Y.append(self.lattice_Y[point.x, point.y])

        vel_x = [self.velx[p.vel] for p in self.points]
        vel_y = [self.vely[p.vel] for p in self.points]
        if self.plot:
            self.quiver = self.ax1.quiver(X, Y, vel_x, vel_y,
                                          units='xy', angles='xy', color='k')

    def plot_order_param(self, num):
        # nwidth = 20
        self.order_param.append(self.cal_order_param())
        self.num += 1
        if self.plot:
            nl = max(self.num - 20, 0)
            nr = 1.25 * 20 + nl
            self.ax2.set_xlim([nl, nr])
            self.l.set_data(np.arange(nl, self.num), self.order_param[nl:])

    def cal_order_param(self):
        # return order parameter
        velx = sum([self.velx[p.vel] for p in self.points])
        vely = sum([self.vely[p.vel] for p in self.points])
        return la.norm([velx, vely]) / self.N

if __name__ == '__main__':
    main = Main(Lx=40, Ly=40, rho=0.9, T=0.41, frames=300, plot=True)
    # main = Main(Lx=40, Ly=40, T=0.6, frames=1000, plot=True)
