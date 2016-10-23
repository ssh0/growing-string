#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto


from growing_string import Main
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


class FillBucket(object):
    def __init__(self, main, plot_type='fill'):
        self.lattice = main.lattice
        self.lattice_X = main.lattice_X
        self.lattice_Y = main.lattice_Y
        self.doubled_lattice = np.zeros((self.lattice.Lx * 2, self.lattice.Ly),
                                        dtype=np.bool)
        self.define_kagome_lattice()
        self.string = main.strings[0]
        self.plot_type = plot_type

        doubled_lattice = self.create_doubled_lattice()
        self.doubled_lattice = self.fill_inside(doubled_lattice)

    def create_doubled_lattice(self):
        str_pos = self.string.pos.tolist()
        check_index = [(i, j)
                       for i in range(self.lattice.Lx)
                       for j in range(self.lattice.Ly)
                       if [i, j] in str_pos]

        for i, j in check_index:
            k = str_pos.index([i, j])
            vec = self.string.vec[k]
            if vec in [0, 3]:
                continue

            if vec == 1:
                x = 2 * i
                y = j
            elif vec == 2:
                x = 2 * i - 1
                y = j
            elif vec == 4:
                x = 2 * i
                y = j - 1
            elif vec == 5:
                x = 2 * i + 1
                y =  j - 1

            self.doubled_lattice[x, y] = True
        return self.doubled_lattice

    def fill_inside(self, arr):
        """Fill inside

        arr: (m x n: boolean ndarray)
        """
        size_x, size_y = arr.shape
        ret_arr = np.zeros((size_x, size_y), dtype=np.bool)
        for j in range(size_y):
            flag = False
            for i in range(size_x):
                tf = arr[i, j]
                if flag ^ tf:
                    ret_arr[i, j] = True

                if tf:
                    flag = not flag
        return ret_arr

    def define_kagome_lattice(self):
        size_x, size_y = self.lattice.Lx, self.lattice.Ly
        x_even = self.lattice_X + 0.5 * self.lattice.dx
        y_even = self.lattice_Y + self.lattice.dy / 3.
        x_odd = np.roll(self.lattice_X, -1, axis=0)
        y_odd = np.roll(self.lattice_Y, -1, axis=0) + (2 * self.lattice.dy) / 3.
        self.kagome_X = np.hstack((x_even, x_odd)).reshape(2 * size_x, size_y)
        self.kagome_Y = np.hstack((y_even, y_odd)).reshape(2 * size_x, size_y)

    def plot_all(self, plot_type=None):
        if plot_type is None:
            plot_type = self.plot_type

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

        triang = tri.Triangulation(lattice_X, lattice_Y)
        self.ax.triplot(triang, color='#d5d5d5', marker='.', markersize=1)

        self.lines = [self.ax.plot([], [], linestyle='-',
                                   color='black',
                                   markerfacecolor='black',
                                   markeredgecolor='black')[0]
                      for i in range(self.lattice.Lx)]

        i = 0
        s = self.string
        start = 0
        for j, pos1, pos2 in zip(range(len(s.pos) - 1), s.pos[:-1], s.pos[1:]):
            dist_x = abs(self.lattice_X[pos1[0], pos1[1]] -
                        self.lattice_X[pos2[0], pos2[1]])
            dist_y = abs(self.lattice_Y[pos1[0], pos1[1]] -
                        self.lattice_Y[pos2[0], pos2[1]])
            if dist_x > 1.5 * self.lattice.dx or dist_y > 1.5 * self.lattice.dy:
                x = s.pos_x[start:j + 1]
                y = s.pos_y[start:j + 1]
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

        dx = self.lattice.dx
        dy = self.lattice.dy
        if plot_type == 'fill':
            X = [self.lattice_X[_x, _y] for _x, _y in zip(s.pos_x, s.pos_y)]
            Y = [self.lattice_Y[_x, _y] for _x, _y in zip(s.pos_x, s.pos_y)]
            patches = [Polygon(np.array([X, Y]).T.tolist())]
            p = PatchCollection(patches, color='green')
            self.ax.add_collection(p)
        elif plot_type == 'point':
            # # plot by Point
            index = np.where(self.doubled_lattice)
            X = self.kagome_X[index]
            Y = self.kagome_Y[index]
            self.ax.plot(X, Y, 'r.', alpha=0.5)
        plt.show()


if __name__ == '__main__':
    L = 60
    frames = 1000

    params = {
        'Lx': L,
        'Ly': L,
        'frames': frames,
        'beta': 0.,
        'weight_const': 0.,
        # 'boundary': {'h': 'periodic', 'v': 'periodic'},
        'boundary': {'h': 'reflective', 'v': 'reflective'},
        'plot': False,
        'plot_surface': False,
        'interval': 0,
    }

    # loop
    # main = Main(strings=[{'id': 1, 'x': L / 4, 'y': L / 2, 'vec': [0, 4, 2]}],
    #             **params
    #             )
    main = Main(strings=[{'id': 1, 'x': L / 2, 'y': L / 2, 'vec': [0, 4, 2]}],
                **params
                )
    # bucket = FillBucket(main, plot_type='fill')
    bucket = FillBucket(main)
    # bucket.plot_all(plot_type='point')
    bucket.plot_all(plot_type='fill')

