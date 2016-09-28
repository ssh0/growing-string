#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-06-09

from growing_string import Main

if __name__ == '__main__':
    # reflective
    main = Main(Lx=60, Ly=60, size=[60,] * 1,
                strings=[{'id': 1, 'x': 0, 'y': 10, 'vec': [0] * 60}],
                boundary={'h': 'periodic', 'v': 'reflective'})

