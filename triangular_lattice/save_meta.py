#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-11-17

import json
import time


def save(base, after='', timestamp=True, **kwargs):
    if timestamp:
        current_time = time.strftime("%y%m%d_%H%M%S")
    else:
        current_time = ''
    fpath = base + current_time + after + '.json'
    with open(fpath, 'w') as f:
        json.dump(kwargs, f, sort_keys=True, indent=2)

    print "[saved] {}".format(fpath)


if __name__ == '__main__':
    params = {
        'num_of_strings': 30,
        'L': 2000,
        'frames': 1000,
        'plot': False,
        'optimize': False,
        'save_image': False,
        'save_data': True,
    }

    save('./', **params)
