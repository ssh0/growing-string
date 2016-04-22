#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto


class Euler(object):

    def __init__(self, function):
        """ Initialize function."""
        self.function = function

    def solve(self, y, t, h):
        """ Solve the system ODEs.

        --- arguments ---
        y: Array of initial values (ndarray)
        t: Time (float)
        h: Stepsize (float)
        """
        return y + h * self.function(t, y)
