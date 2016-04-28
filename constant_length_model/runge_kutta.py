#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# written by Shotaro Fujimoto


class RK4(object):

    def __init__(self, function):
        """ Initialize function"""
        self.function = function

    def solve(self, y, t, h):
        """ Solve the system ODEs

        --- arguments ---
        y: A list of initial values (ndarray)
        t: Time (float)
        h: Stepsize (float)
        """
        k1 = h * self.function(t, y)
        k2 = h * self.function(t + h / 2, y + h * k1 / 2)
        k3 = h * self.function(t + h / 2, y + h * k2 / 2)
        k4 = h * self.function(t + h, y + h * k3 / 2)

        return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
