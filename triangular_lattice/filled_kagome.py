#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-10-07


from fill_bucket import FillBucket
from growing_string import Main
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri


class FilledKagome:
    def __init__(self, L=60, frames=1000, beta=0.):
        fill = self.init(L, frames, beta)
        self.lattice = fill.doubled_lattice
        self.lattice_X, self.lattice_Y = fill.kagome_X, fill.kagome_Y
        self.R = self.calc_radius_of_rotation(self.lattice)

    def init(self, L, frames, beta):
        params = {
            'Lx': L,
            'Ly': L,
            'frames': frames,
            'beta': beta,
            'weight_const': 0.5,
            'strings': [{'id': 1, 'x': L / 2, 'y': L / 2, 'vec': [0, 4, 2]}, ],
            'boundary': {'h': 'reflective', 'v': 'reflective'},
            'plot': False,
            'plot_surface': False,
            'interval': 1,
        }

        self.main = Main(**params)
        fill = FillBucket(self.main)
        return fill

    def calc_radius_of_rotation(self, lattice):
        """Calcurate the raidus of rotation of the giving lattice.

        lattice: M x N ndarray
        """
        # cast
        lattice = lattice.astype(bool)
        # get index lists which is True
        pos = np.where(lattice)
        N = len(pos[0])
        x = self.lattice_X[pos]
        y = self.lattice_Y[pos]
        self.X0 = np.average(x)
        self.Y0 = np.average(y)
        r = np.sqrt(np.sum((x - self.X0) ** 2 + (y - self.Y0) ** 2) / N)
        return r

    def plot_all(self):
        self.fig, self.ax = plt.subplots(figsize=(8, 8))

        lattice_X = self.main.lattice.coordinates_x
        lattice_Y = self.main.lattice.coordinates_y
        X_min, X_max = min(lattice_X) - 0.1, max(lattice_X) + 0.1
        Y_min, Y_max = min(lattice_Y) - 0.1, max(lattice_Y) + 0.1
        self.ax.set_xlim([X_min, X_max])
        self.ax.set_ylim([Y_min, Y_max])
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.set_aspect('equal')

        triang = tri.Triangulation(lattice_X, lattice_Y)
        self.ax.triplot(triang, color='#d5d5d5', marker='.', markersize=1)

        # # plot by Point
        index = np.where(self.lattice)
        X = self.lattice_X[index]
        Y = self.lattice_Y[index]
        self.ax.plot(X, Y, 'r.')

        rad = np.linspace(0., 2. * np.pi, num=100)
        x_circ = self.R * np.cos(rad) + self.X0
        y_circ = self.R * np.sin(rad) + self.Y0
        self.ax.plot(self.X0, self.Y0, 'go', alpha=0.5)
        self.ax.plot(x_circ, y_circ, 'g-', alpha=0.5)

        plt.show()


if __name__ == '__main__':
    setup = {
        'L': 500,
        'frames': 600,
        'beta': 20.
    }

    # Simple Simulation
    filled_kagome = FilledKagome(**setup)
    filled_kagome.plot_all()
    print filled_kagome.R

