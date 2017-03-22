#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2017-03-22
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.tri as tri
import matplotlib.pyplot as plt
import numpy as np
from triangular import LatticeTriangular as LT
from strings import String
from growing_string import Main
from triangular_random import randomize


class Sim(Main):
    def __init__(self, Lx=40, Ly=40,
                 boundary={'h': 'periodic', 'v': 'periodic'},
                 size=[5, 4, 10, 12],
                 plot=True,
                 plot_surface=True,
                 save_image=False,
                 save_video=False,
                 filename_image="",
                 filename_video="",
                 frames=1000,
                 beta = 2.,
                 interval=1,
                 weight_const=0.5,
                 strings=None,
                 pre_function=None,
                 post_function=None):

        self.lattice = LT(
            np.zeros((Lx, Ly), dtype=np.int),
            scale=float(max(Lx, Ly)),
            boundary=boundary
        )

        ## randomize
        randomize(self.lattice)

        ## if the lattice size exceeds 200, don't draw triangular lattice.
        if max(self.lattice.Lx, self.lattice.Ly) < 200:
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
        self.number_of_lines = Lx

        if strings is None:
            # Put the strings to the lattice
            self.strings = self.create_random_strings(len(size), size)
        else:
            self.strings = [String(self.lattice, **st) for st in strings]
        for string in self.strings:
            self.occupied[string.pos_x, string.pos_y] = True


        self.plot = plot
        self.plot_surface = plot_surface
        self.save_image = save_image
        self.save_video = save_video
        if self.save_image:
            if filename_image == "":
                raise AttributeError("`filename_image` is empty.")
            else:
                self.filename_image = filename_image

        if self.save_video:
            if self.plot:
                raise AttributeError("`save` and `plot` method can't be set both True.")
            if filename_video == "":
                raise AttributeError("`filename_video` is empty.")
            else:
                self.filename_video = filename_video


        self.interval = interval
        self.frames = frames

        # 逆温度
        self.beta = beta
        self.weight_const = weight_const

        self.bonding_pairs = {i: {} for i in range(len(self.strings))}
        for key in self.bonding_pairs.keys():
            value = self.get_bonding_pairs(
                s=self.strings[key],
                # indexes=[[0, len(self.strings[key].pos)]]
                index_start=0,
                index_stop=len(self.strings[key].pos)
            )

            self.bonding_pairs[key] = value

        self.pre_function = pre_function
        self.post_function = post_function
        self.pre_func_res = []
        self.post_func_res = []

        # Plot triangular-lattice points, string on it, and so on
        if self.plot:
            self.plot_all()
            self.start_animation()
        elif self.save_video:
            self.plot_all()
            self.start_animation(filename=self.filename_video)
        else:
            t = 0
            while t < self.frames:
                try:
                    self.update()
                    t += 1
                except StopIteration:
                    break

        if self.save_image:
            if not self.__dict__.has_key('fig'):
                self.plot_all()
            self.fig.savefig(self.filename_image)
            plt.close()
            # print("Image file is successfully saved at '%s'." % filename_image)

    def plot_all(self):
        """軸の設定，三角格子の描画，線分描画要素の用意などを行う

        ここからFuncAnimationを使ってアニメーション表示を行うようにする
        """
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

        ## if the lattice size exceeds 200, don't draw triangular lattice.
        if max(self.lattice.Lx, self.lattice.Ly) < 200:
            self.ax.triplot(self.triang_random, color='#d5d5d5', lw=0.5)


        self.lines = [self.ax.plot([], [], linestyle='-',
                                   color='black',
                                   markerfacecolor='black',
                                   markeredgecolor='black')[0]
                      for i in range(self.number_of_lines)]
        if self.plot_surface:
            # self.__num_surface = 1
            # self.lines.append(self.ax.plot([], [], '.', color='#ff0000')[0])
            self.__num_surface = 9
            self.lines += [self.ax.plot([], [], '.',
                                        )[0]
                           for i in range(self.__num_surface)]
        self.plot_string()


if __name__ == '__main__':
    L = 100
    params = {
        'Lx': L,
        'Ly': L,
        'size': [3,] * 1, 
        'frames': 1000,
        'beta': 2.,
        'plot': True,
        'plot_surface': False,
        'interval': 0,
        'strings': [{'id': 1, 'x': L/4, 'y': L/2, 'vec': [0, 4]}]
        # 'strings': [{'id': 1, 'x': L/4, 'y': L/2, 'vec': [0, 4, 2]}]
    }

    main= Sim(**params)
