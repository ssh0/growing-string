#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2017-01-23

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector


class SpanFitting:
    """Fitting (x, y) data interactively"""
    def __init__(self, ax, x, y, optimizer, parameters,
                 attacth_text=True, attacth_label=False):
        """
        === Arguments ===
        ax: matplotlib.Axes
        x, y: raw data (1-d array)
        optimizer: Optimizer class
                   --- self.func(parameters, x, y): calc residual(= y - f(x))
                   --- self.fitting(): return fitted parameters
                   --- self.fitted(x): return f(x) with fitted patramere
        parameters: (list) used for optimizer
        attacth_text: (bool) fitted paramter will be shown near the fitted curve
        attacth_label: (bool) fitted paramter will be shown in a legend
        """
        self.ax = ax
        xscale = self.ax.get_xscale()
        yscale = self.ax.get_yscale()
        if yscale == 'linear':
            if xscale == 'linear':
                self.plot = self.ax.plot
            elif xscale == 'log':
                self.plot = self.ax.semilogx
        elif yscale == 'log':
            if xscale == 'linear':
                self.plot = self.ax.semilogy
            elif xscale == 'log':
                self.plot = self.ax.loglog

        self.x, self.y = x, y
        self.optimizer = optimizer
        self.parameters = parameters
        self.attach_text = attacth_text
        self.attach_label = attacth_label
        self.height_span = height_span
        self.ln = None

    def onselect(self, vmin, vmax):
        if self.ln is not None:
            self.ln.remove()
            if self.attach_text:
                self.text.remove()

        selected_index = np.where((self.x >= vmin) & (self.x <= vmax))
        optimizer = self.optimizer(
            args=(self.x[selected_index], self.y[selected_index]),
            parameters=self.parameters)
        result = optimizer.fitting()
        result_text = ', '.join(['{} = {}'.format(k, v)
                                 for k, v in result.items()])
        print(result_text)
        X = self.x[selected_index]
        Y = optimizer.fitted(X)
        if self.attach_label:
            self.ln, = self.plot(X, Y, ls='-', label=result_text,
                                 marker='', color='k')
            self.ax.legend(loc='best')
        else:
            self.ln, = self.plot(X, Y, ls='-', marker='', color='k')
        if self.attach_text:
            self.text = self.ax.text(
                (X[0] + X[-1]) / 2., (Y[0] + Y[-1]) / 2. + self.height_span,
                result_text, ha='center', va='bottom'
            )

    def start(self):
        self.ax.plot(self.x, self.y, '.')
        span = SpanSelector(self.ax, self.onselect, direction='horizontal')
        plt.show()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from optimize import Optimize_powerlaw, Optimize_linear

    x = np.logspace(1, 10, base=2.)
    y = x ** (2 + 0.1 * (np.random.rand() - 0.5))

    fig, ax = plt.subplots()
    ax.set_title('test')
    ax.set_xscale('log')
    ax.set_yscale('log')
    spanfitting = SpanFitting(ax, x, y, Optimize_powerlaw, [0.5, 2])
    spanfitting.start()
