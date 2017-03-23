#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2017-03-23

from growing import Sim
import time


if __name__ == '__main__':
    L = 100
    frames = 2000

    params = {
        'Lx': L,
        'Ly': L,
        'frames': frames,
        'beta': 4.,
        'boundary': {'h': 'periodic', 'v': 'periodic'},
        # 'boundary': {'h': 'reflective', 'v': 'reflective'},
        'interval': 0,
        'plot': False,
        'plot_surface': False,
        'save_image': False,
        'save_video': True,
        'filename_video': "../results/video/random_lattice/" + \
        time.strftime("%y%m%d_%H%M%S") + ".mp4",
    }

    main = Sim(strings=[{'id': 1, 'x': L / 4, 'y': L / 2, 'vec': [0, 4]}],
               **params
               )
