#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-09-27

from growing_string import Main
import random

if __name__ == '__main__':
    # Ly = 40
    # main = Main(Lx=100, Ly=Ly, size=[Ly])

    # Ly = 5
    # main = Main(Lx=10, Ly=Ly, size=[Ly])

    # main = Main(Lx=6, Ly=6, size=[random.randint(4, 12)] * 1, plot=False)
    # main = Main(Lx=50, Ly=50, size=[random.randint(4, 12)] * 1)
    # main = Main(Lx=30, Ly=30, size=[random.randint(4, 12) for i in range(3)])

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


    params = {
        'weight_const': 10.,
        'beta': 1.,
        'boundary': {'h': 'periodic', 'v': 'periodic'},
        'plot': True,
        'plot_surface': True
    }

    # main = Main(Lx=60, Ly=60, size=[3,] * 1,
    #             strings=[{'id': 1, 'x': 30, 'y': 15, 'vec': [0, 4]}],
    #             **params
    #             )

    # main = Main(Lx=120, Ly=120, size=[3,] * 1,
    #             strings=[{'id': 1, 'x': 60, 'y': 30, 'vec': [0, 4]}],
    #             frames=4000,
    #             **params
    #             )

    # print main.post_func_res

    # main = Main(Lx=10, Ly=10, size=[3,] * 1,
    #             strings=[{'id': 1, 'x': 5, 'y': 2, 'vec': [0, 4]}])

    main = Main(Lx=60, Ly=60, size=[4,] * 1,
                strings=[{'id': 1, 'x': 30, 'y': 15, 'vec': [0, 4, 2]}],
                **params
                )
