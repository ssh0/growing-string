#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-06-03

import numpy as np


class String:

    def __init__(self, lattice, id, x, y, vec):
        self.lattice = lattice
        self.id = id
        self.x, self.y = x, y
        self.vec = vec
        self.update_pos()

    def update_pos(self):
        self.pos_x, self.pos_y = [self.x], [self.y]
        for i in self.vec:
            newx, newy = self.lattice.neighborhoods[self.pos_x[-1],
                                                    self.pos_y[-1]]
            self.pos_x.append(newx[i])
            self.pos_y.append(newy[i])

        self.pos_x = np.array(self.pos_x)
        self.pos_y = np.array(self.pos_y)
        self.pos = np.array([self.pos_x, self.pos_y]).T

    def append(self, x):
        self.vec.append(x)
        self.update_pos()

    def insert(self, i, x):
        self.vec.insert(i, x)
        self.update_pos()

    def putright(self, x):
        self.vec = [x] + self.vec[:-1]
        self.update_pos()

    def update_vec(self, vec):
        self.vec = vec
        self.update_pos()

    def follow(self, X):
        x, y, vec = X
        self.x, self.y = x, y
        rmx, rmy = self.pos_x[-1], self.pos_y[-1]
        self.vec = [vec] + self.vec[:-1]
        self.update_pos()
        return rmx, rmy

    def update_starting_point(self, x, y):
        self.x, self.y = x, y
        self.update_pos()
