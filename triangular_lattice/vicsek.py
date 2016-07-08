#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-05-15

from triangular import LatticeTriangular as LT
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.animation as animation
import numpy as np
from numpy import linalg as la
import random
import time


class Point:

    def __init__(self, id, ix, iy):
        self.id = id
        self.x = ix
        self.y = iy
        # vel is unified and the value of it implies the direction of the
        # velocity
        self.vel = np.random.randint(0, 5)
        self.priority = np.random.rand()


class Points:

    def __init__(self, N, Lx, Ly):
        self.points = []
        for n in range(N):
            ix = random.randint(0, Lx - 1)
            iy = random.randint(0, Ly - 1)
            self.points.append(Point(n, ix, iy))


class Main:

    def __init__(self, Lx=20, Ly=20, rho=0.9, lattice_scale=10, T=0.4, plot=True,
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

        self.lattice_X = self.lattice.coordinates_x
        self.lattice_Y = self.lattice.coordinates_y
        self.lattice_X = np.array(self.lattice_X).reshape(Lx, Ly)
        self.lattice_Y = np.array(self.lattice_Y).reshape(Lx, Ly)
        X_min, X_max = np.min(self.lattice_X), np.max(self.lattice_X)
        Y_min, Y_max = np.min(self.lattice_Y), np.max(self.lattice_Y)

        X, Y = [], []
        vel_x, vel_y = [], []
        for point in self.points:
            X.append(self.lattice_X[point.x, point.y])
            Y.append(self.lattice_Y[point.x, point.y])
            angle = point.vel * np.pi / 3.
            vel_x.append(np.cos(angle))
            vel_y.append(- np.sin(angle))

        if self.plot:
            self.fig, (self.ax1, self.ax2) = plt.subplots(
                2, 1, figsize=(8, 10))
            self.ax1.set_xlim([X_min, X_max])
            self.ax1.set_ylim([Y_min, Y_max])
            self.ax1.set_xticklabels([])
            self.ax1.set_yticklabels([])
            self.ax1.set_title("Lattice-Gas model for collective motion")
            self.triang = tri.Triangulation(self.lattice_X.flatten(),
                                            self.lattice_Y.flatten())
            self.ax1.triplot(self.triang, color='whitesmoke', marker='o',
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
            nnx, nny = self.lattice.neighborhoods[point.x, point.y]
            for x, y in zip(nnx, nny):
                if lowest.has_key((x, y)):
                    angle = lowest[(x, y)].vel * np.pi / 3.
                    velocities += np.array([np.cos(angle), np.sin(angle)])

            # 可能な6つの速度ベクトルとの内積を計算
            u_alpha = [i * np.pi / 3. for i in range(6)]
            u_alpha = [np.array([np.cos(ang), np.sin(ang)]) for ang in u_alpha]

            # ボルツマン分布に従って確率的に方向を決定
            A = [np.exp(self.beta * np.dot(u, velocities)) for u in u_alpha]
            rand = random.random() * sum(A)
            p = 0
            for i, P in enumerate(A):
                p += P
                if rand < p:
                    point.vel = i
                    break

        # それ以外はランダムに向きを変えるように
        for point in upper:
            # change the velocity of the point
            point.vel = random.randint(0, 5)

        # 各点の座標とベクトルを更新し，描画
        self.update_quivers()

        # オーダーパラメーターをプロット
        self.plot_order_param(num)

        return self.quiver, self.l

    def update_quivers(self):
        # Get information to plot
        X, Y = [], []
        vel_x, vel_y = [], []
        for point in self.points:
            # Get possible direction
            newx, newy = self.lattice.neighborhoods[point.x, point.y]
            # Choose one by its velocity
            point.x, point.y = newx[point.vel], newy[point.vel]
            X.append(self.lattice_X[point.x, point.y])
            Y.append(self.lattice_Y[point.x, point.y])

            angle = point.vel * np.pi / 3.
            vel_x.append(np.cos(angle))
            vel_y.append(- np.sin(angle))
        if self.plot:
            self.quiver = self.ax1.quiver(X, Y, vel_x, vel_y,
                                          units='xy', angles='xy', color='k')

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
    main = Main(Lx=40, Ly=40, rho=0.9, T=0.41, frames=300, plot=True)
    # main = Main(Lx=40, Ly=40, T=0.6, frames=1000, plot=True)
