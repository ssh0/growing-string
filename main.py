#!/usr/bin/env python
# -*- coding:utf-8 -*-
""" Numerical simulation of growing strings

各点間をバネでつなぎ、その自然長が時間と共に増大することを考える。
この自然長、もしくは2点間の距離がある閾値を超えた時、新たに2点間に点を置く。
点に作用する力は、バネによる力と、曲げ弾性による力、粘性による摩擦力である。
オイラー法または4次のルンゲクッタ法でこれを解き、matplotlibでアニメーションに
して表示する。
"""
__author__ = "Shotaro Fujimoto"
__date__ = "2016/4/12"

from euler import Euler
from runge_kutta import RK4
from points import Points
from runner import String_Simulation


if __name__ == '__main__':
    import argparse
    import numpy as np
    parser = argparse.ArgumentParser(
        description='Simulation of growing strings')
    parser.add_argument('--debug', dest='debug_mode', action='store_true',
                        help='debug mode: print some verbose information')
    args = parser.parse_args()

    # initial values
    params = {
        "open 4": {
            'x': np.array([-2., 0., 2., 4.]),
            'y': np.array([0., -0.01, 0.01, 0.]),
            'nl': np.array([2., 2., 2.]),
            'K': np.array([300., 300., 300.]),
            'length_limit': 4.,
        },

        "close 4": {
            'x': np.array([0., 1., 1., 0.]),
            'y': np.array([0., 0., 1., 1.]),
            'nl': np.array([1., 1., 1., 1.]),
            'K': np.array([300., 300., 300., 300.]),
            'length_limit': 4.,
        },

        "close 3": {
            'x': np.array([0., 1.73, 1.73]),
            'y': np.array([0., 1., -1.]),
            'nl': np.array([2., 2., 2.]),
            'K': np.array([300., 300., 300.]),
            'length_limit': 4.,
        },
    }

    x = np.arange(-5., 5., step=1.)
    params.update({
        "open many": {
            'x': x,
            'y': np.array([0. + 0.1 * (random.random() - 0.5) for n in
                           range(len(x))]),
            'nl': np.array([1.] * (len(x) - 1)),
            'K': np.array([15.] * (len(x) - 1)),
            'length_limit': 4.,
        }
    })

    # common parameters (overwrite)
    for k in params.iterkeys():
        params_after = {
            'h': 0.005,
            't_max': 300.,
            'm': 3.,
            'e': 30.,
            'D': 10.,
            'debug_mode': args.debug_mode,
            'self_avoiding': True
        }
        for kk in params_after.iterkeys():
            if kk in params[k]:
                params_after[kk] = params[k][kk]
        params[k].update(params_after)

    sim = String_Simulation(params['open many'])
    sim.run()
