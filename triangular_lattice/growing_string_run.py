#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-09-27

from growing_string import Main
import time

if __name__ == '__main__':
    # import timeit
    # print(timeit.timeit("Main(Lx=60, Ly=60, size=[3,] * 1, \
    #                           strings=[{'id': 1, 'x': 15, 'y': 30, 'vec': [0, 4]}], \
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

    # L = 10
    # frames = 20
    L = 1500
    frames = 1000
    # frames = 100 * 100

    params = {
        'Lx': L,
        'Ly': L,
        'frames': frames,
        'beta': 12.,
        'weight_const': 0.4,
        'boundary': {'h': 'periodic', 'v': 'periodic'},
        # 'boundary': {'h': 'reflective', 'v': 'reflective'},
        'plot': True,
        'plot_surface': True,
        'interval': 1,
    }

    # save img
    # params.update({
    #     'plot': False,
    #     'plot_surface': False,
    #     'save_image': True,
    #     'filename_image': "results/img/growing_string_" + time.strftime("%y%m%d_%H%M%S"),
    # })

    # save video
    params.update({
        'plot': False,
        'plot_surface': False,
        'save_image': False,
        'save_video': True,
        'filename_video': "results/video/growing_string_" + time.strftime("%y%m%d_%H%M%S") + ".mp4",
    })

    # save img and video
    # params.update({
    #     'plot': False,
    #     'save_image': True,
    #     'save_video': True,
    #     'filename_image': "results/img/growing_string_" + time.strftime("%y%m%d_%H%M%S"),
    #     'filename_video': "results/video/growing_string_" + time.strftime("%y%m%d_%H%M%S"),
    # })


    # # open
    main = Main(strings=[{'id': 1, 'x': L / 4, 'y': L / 2, 'vec': [0, 4]}],
                **params
                )

    # loop
    # main = Main(strings=[{'id': 1, 'x': L / 4, 'y': L / 2, 'vec': [0, 4, 2]}],
    #             **params
    #             )

