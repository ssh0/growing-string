#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-11-17

import numpy as np
import time


def save(base, after='', timestamp=True, **kwargs):
    if timestamp:
        current_time = time.strftime("%y%m%d_%H%M%S")
    else:
        current_time = ''
    fpath = base + current_time + after + '.npz'
    np.savez(fpath, **kwargs)
    print "[saved] {}".format(fpath)


if __name__ == '__main__':
    params = {
        'Ls': [0, 1, 2, 3],
        'f': [3, 1, 6, 1],
    }

    save('./', **params)

