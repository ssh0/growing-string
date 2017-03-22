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
from strings import String
from base import Main as Base
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


class Main(Base):
    def __init__(self, Lx=40, Ly=40, N=4, size=[5, 4, 10, 12], interval=50,
                 plot=True):
        # Create triangular lattice with given parameters
        self.lattice = LT(np.zeros((Lx, Ly), dtype=np.int),
                          scale=float(max(Lx, Ly)),
                          boundary={'h': 'periodic', 'v': 'periodic'})
        ## randomize
        randomize(self.lattice)

        self.triang_standard = tri.Triangulation(self.lattice.coordinates_x,
                                                 self.lattice.coordinates_y)
        self.triang_random = tri.Triangulation(
            self.lattice.coordinates_x,
            self.lattice.coordinates_y,
            triangles=self.triang_standard.triangles)

        self.occupied = np.zeros((Lx, Ly), dtype=np.bool)
        self.number_of_lines = Lx

        # Put the strings to the lattice
        self.strings = self.create_random_strings(N, size)

        self.plot = plot
        self.interval = interval

    def plot_all(self):
        """軸の設定，三角格子の描画，線分描画要素の用意などを行う

        ここからFuncAnimationを使ってアニメーション表示を行うようにする
        """
        if self.__dict__.has_key('frames'):
            frames = self.frames
        else:
            frames = 1000
        self.fig, self.ax = plt.subplots(figsize=(8, 8))

        lattice_X = self.lattice.coordinates_x
        lattice_Y = self.lattice.coordinates_y
        X_min, X_max = min(lattice_X) - 0.1, max(lattice_X) + 0.1
        Y_min, Y_max = min(lattice_Y) - 0.1, max(lattice_Y) + 0.1
        self.ax.set_xlim([X_min, X_max])
        self.ax.set_ylim([Y_min, Y_max])
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.set_aspect('equal')

        self.ax.triplot(self.triang_random, color='#d5d5d5', lw=0.5)

        self.lines = [self.ax.plot([], [], marker='.', linestyle='-',
                                   color='black',
                                   markerfacecolor='black',
                                   markeredgecolor='black')[0]
                      for i in range(self.number_of_lines)]

        self.lattice_X = self.lattice.coordinates_x.reshape(self.lattice.Lx,
                                                            self.lattice.Ly)
        self.lattice_Y = self.lattice.coordinates_y.reshape(self.lattice.Lx,
                                                            self.lattice.Ly)
        self.plot_string()

        def init_func(*arg):
            return self.lines

        ani = animation.FuncAnimation(self.fig, self.update, frames=frames,
                                      init_func=init_func,
                                      interval=self.interval,
                                      blit=True, repeat=False)
        plt.show()


if __name__ == '__main__':
    main = Main()
    # Plot triangular-lattice points, string on it, and so on
    if main.plot:
        main.plot_all()
    else:
        t = 0
        while t < main.frames:
            try:
                main.update()
                t += 1
            except StopIteration:
                break
