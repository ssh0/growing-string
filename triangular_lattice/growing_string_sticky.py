#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-06-09

from growing_string import Main
import numpy as np


class Growing_sticky(Main):
    def __init__(self, Lx, Ly):
        Main.__init__(self, Lx=Lx, Ly=Ly, plot=True,
                      frames=1000,
                      strings=[{'id': 1, 'x': int(Lx / 2),
                                'y': - int(Lx / 4) % Ly,
                                'vec': [0] * (Ly - 1)}]
                     )


if __name__ == '__main__':
    main = Growing_sticky(Lx=50, Ly=20)
