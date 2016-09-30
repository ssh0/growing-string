#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-05-30

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


class Points:
    def __init__(self, N, Lx, Ly):
        self.points = []
        occupied = [True] * N + [False] * (Lx * Ly - N)
        random.shuffle(occupied)
        occupied = np.array([occupied]).reshape(Lx, Ly)
        n = 0
        for ix, iy in zip(np.where(occupied)[0], np.where(occupied)[1]):
            self.points.append(Point(n, ix, iy))
            n += 1


class Main:

    def __init__(self, Lx=6, Ly=6, rho=0.9, lattice_scale=10, T=0.4, plot=True,
                 frames=100):
        self.lattice = LT(- np.ones((Lx, Ly), dtype=np.int),
                          scale=lattice_scale)
        self.N = int(Lx * Ly * rho)
        self.points = Points(self.N, Lx, Ly)
        self.points = self.points.points
        self.T = T
        self.plot = plot
        self.beta = 1. / self.T
        self.order_param = []
        self.num = 0
        self.lattice_point = {}
        for point in self.points:
            if not self.lattice_point.has_key((point.x, point.y)):
                self.lattice_point[(point.x, point.y)] = point
            else:
                raise UserWarning()

        self.lattice_X = self.lattice.coordinates_x
        self.lattice_Y = self.lattice.coordinates_y
        self.lattice_X = np.array(self.lattice_X).reshape(Lx, Ly)
        self.lattice_Y = np.array(self.lattice_Y).reshape(Lx, Ly)
        X_min, X_max = np.min(self.lattice_X), np.max(self.lattice_X)
        Y_min, Y_max = np.min(self.lattice_Y), np.max(self.lattice_Y)

        self.X, self.Y = [], []
        self.vel_x, self.vel_y = [], []
        for point in self.points:
            self.X.append(self.lattice_X[point.x, point.y])
            self.Y.append(self.lattice_Y[point.x, point.y])
            angle = point.vel * np.pi / 3.
            self.vel_x.append(np.cos(angle))
            self.vel_y.append(- np.sin(angle))

        if self.plot:
            self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(8, 10))
            self.ax1.set_xlim([X_min, X_max])
            self.ax1.set_ylim([Y_min, Y_max])
            self.ax1.set_xticklabels([])
            self.ax1.set_yticklabels([])
            self.ax1.set_aspect('equal')
            self.ax1.set_title("Lattice-Gas model for collective motion")
            self.triang = tri.Triangulation(self.lattice_X.flatten(),
                                            self.lattice_Y.flatten())
            self.ax1.triplot(self.triang, color='whitesmoke', marker='.',
                             markersize=1)
            self.l, = self.ax2.plot([], [], 'b-')
            self.ax2.set_title(r"Order parameter $m=\frac{1}{N} |\sum \vec{u}_{i}|$ ($T = %.2f$)"
                               % self.T)
            self.ax2.set_ylim([0, 1.])
            ani = animation.FuncAnimation(self.fig, self.update, frames=frames,
                                          interval=1, blit=True, repeat=False)
            plt.show()
        else:
            for i in range(100):
                self.update(i)
            # print self.order_param[-1]

    def update(self, num):
        max_repeat = 10
        i = 0
        while True:
            # 一つのPointを選択
            p = random.choice(self.points)

            # 速度方向が空いているかどうか
            nnx, nny = self.lattice.neighborhoods[p.x, p.y]

            # 占有されていないもののみを抽出
            not_occupied = {}
            for v, x, y in zip(range(len(nnx)), nnx, nny):
                if not self.lattice_point.has_key((x, y)):
                    not_occupied[v] = (x, y)

            if len(not_occupied) == 0:
                # 点を選び直し
                i += 1
                continue
            elif self.lattice_point.has_key((nnx[p.vel], nny[p.vel])):
                # ランダムに選択する
                p.vel = random.choice(not_occupied.keys())

            # 点を動かす
            del self.lattice_point[(p.x, p.y)]
            p.x, p.y = nnx[p.vel], nny[p.vel]
            self.lattice_point[(p.x, p.y)] = p
            self.X[p.id] = self.lattice_X[p.x, p.y]
            self.Y[p.id] = self.lattice_Y[p.x, p.y]

            # 最近接効果(decided by Boltzmann eq)をうけて速度変化
            # 最近接の速度の合計を求める
            velocities = np.array([0., 0.])
            nnx, nny = self.lattice.neighborhoods[p.x, p.y]
            not_occupied = {}
            for v, x, y in zip(range(len(nnx)), nnx, nny):
                if self.lattice_point.has_key((x, y)):
                    angle = self.lattice_point[(x, y)].vel * np.pi / 3.
                    velocities += np.array([np.cos(angle), np.sin(angle)])
                else:
                    # 占有されていないもののみを抽出
                    not_occupied[v] = (x, y)

            # ボルツマン分布に従って確率的に速度方向を決定
            u_alpha = [i * np.pi / 3. for i in range(6)]
            u_alpha = [np.array([np.cos(ang), np.sin(ang)]) for ang in u_alpha]
            A = [np.exp(self.beta * np.dot(u, velocities)) for u in u_alpha]
            rand = random.random() * sum(A)
            prob = 0
            for i, P in enumerate(A):
                prob += P
                if rand < prob:
                    p.vel = i
                    break

            angle = p.vel * np.pi / 3.
            self.vel_x[p.id] = np.cos(angle)
            self.vel_y[p.id] = - np.sin(angle)
            if self.plot:
                self.quiver = self.ax1.quiver(self.X, self.Y, self.vel_x, self.vel_y,
                                            units='xy', angles='xy', color='k')
            break

        # オーダーパラメーターをプロット
        self.plot_order_param(num)

        return self.quiver, self.l

    def plot_order_param(self, num):
        nwidth = 20
        m = self.cal_order_param()
        self.order_param.append(m)
        self.num += 1
        nl = max(self.num - nwidth, 0)
        nr = 1.25 * nwidth + nl
        xdata = np.arange(nl, self.num)
        # print xdata
        # print self.order_param
        # print len(xdata)
        # print len(self.order_param)
        if self.plot:
            self.ax2.set_xlim([nl, nr])
            self.l.set_data(xdata, self.order_param[nl:])

    def cal_order_param(self):
        # return order parameter
        velocities = np.array([0., 0.])
        for point in self.points:
            angle = point.vel * np.pi / 3.
            velocities += np.array([np.cos(angle), np.sin(angle)])

        m = la.norm(velocities) / self.N
        return m

if __name__ == '__main__':
    main = Main(Lx=10, Ly=10, rho=0.6, T=1.5, frames=1000, plot=True)
    # main = Main(Lx=40, Ly=40, T=0.6, frames=1000, plot=True)

