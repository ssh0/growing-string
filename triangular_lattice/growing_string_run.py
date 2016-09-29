#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-09-27

from growing_string import Main
import random

if __name__ == '__main__':
    # import timeit
    # print(timeit.timeit("Main(Lx=60, Ly=60, size=[3,] * 1, \
    #                           strings=[{'id': 1, 'x': 30, 'y': 15, 'vec': [0, 4]}], \
    #                           plot=False)",
    #                     setup="from __main__ import Main",
    #                     number=10
    #                     ))

    def store_bonding_pairs(self, i, s):
        """Store bonding pairs function.

        self: instance of Main class
        i: (int) num ID of string
        s: (object) instance of String class
        """
        return self.bonding_pairs[i]

    L = 60
    frames = 1000

    params = {
        'Lx': L,
        'Ly': L,
        'frames': frames,
        'beta': 1.,
        'weight_const': 0.,
        'boundary': {'h': 'periodic', 'v': 'periodic'},
        # 'boundary': {'h': 'reflective', 'v': 'reflective'},
        'plot': True,
        'plot_surface': True,
        'interval': 0,
    }

    # # open
    # main = Main(size=[3,] * 1,
    #             strings=[{'id': 1, 'x': L / 2, 'y': L / 4, 'vec': [0, 4]}],
    #             **params
    #             )

    # # loop
    main = Main(size=[4,] * 1,
                strings=[{'id': 1, 'x': L / 2, 'y': L / 4, 'vec': [0, 4, 2]}],
                **params
                )

