#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-08-19

import scipy.optimize as optimize


class Optimize_powerlaw:
    def __init__(self, args, parameters):
        self.args = args
        self.parameters = parameters

    def func(self, parameters, x, y):
        c = parameters[0]
        D = parameters[1]
        residual = y - c * (x ** D)
        return residual

    def fitting(self):
        result = optimize.leastsq(self.func, x0=self.parameters, args=self.args)
        self.c = result[0][0]
        self.D = result[0][1]
        return {'c': self.c, 'D': self.D}

    def fitted(self, x):
        return self.c * (x ** self.D)
