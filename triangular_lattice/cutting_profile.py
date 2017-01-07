#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2017-01-07

from growing_string import Main
from triangular import LatticeTriangular as LT
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from tqdm import tqdm
import time
import random
import save_data as sd


class CuttingProfile(object):
    def __init__(self, frames, beta, c=0.4, L=100, save_result=False,
                 plot_raw_result=False):
        self.L = L
        # self.pbar = tqdm(total=frames)
        self.frames = frames
        self.beta = beta
        self.weight_const = c
        self.t = 1
        self.save_result = save_result
        self.plot_raw_result = plot_raw_result

    def start(self):
        self.main = Main(
            Lx=self.L,
            Ly=self.L,
            size=[3,],
            plot=False,
            plot_surface=False,
            frames=self.frames,
            strings=[{'id': 1, 'x': self.L/4, 'y': self.L/2, 'vec': [0, 4]}],
            beta=self.beta,
            weight_const=self.weight_const,
            # pre_function=self.get_cutting_profiles
        )
        self.cutting_profiles = self.get_cutting_profiles()
        if self.save_result:
            sd.save("results/data/cutting_profile/" +
                    "frames=%d_beta=%2.2f_" % (self.frames, self.beta),
                    beta=self.beta, L=self.L, frames=self.frames,
                    weight_const=self.weight_const,
                    cutting_profiles=self.cutting_profiles
                    )
        if self.plot_raw_result:
            self.plot_result()

    def get_cutting_profiles(self):
        """6つの方向でそれぞれに対して，ベクトルの向きに対して左側に
        閉じている領域要素の位置を記録
        """
        return [self.get_cutting_profile_for(k) for k in range(6)]

    def get_cutting_profile_for(self, k):
        """k方向のベクトルに対して左側に閉じた部分領域の位置を取得"""
        positions = []
        s = self.main.strings[0]
        vec = s.vec
        for i, v in enumerate(vec):
            if v == (k + 5) % 6:
                if i > 0 and vec[i - 1] == (k + 1) % 6:
                    positions.append(s.pos[i])
            elif v == (k + 4) % 6:
                if i > 0 and vec[i - 1] == (k + 2) % 6:
                    positions.append(s.pos[i])
        return positions

    def plot_result(self):
        self.main.plot_all()
        for k in range(6):
            positions = self.cutting_profiles[k]
            X = [self.main.lattice_X[_x, _y] for _x, _y in positions]
            Y = [self.main.lattice_Y[_x, _y] for _x, _y in positions]
            self.main.ax.plot(X, Y, 'o', label='vec: {}'.format(k),
                              color=cm.rainbow(float(k) / 5))

        self.main.ax.legend(loc='best')
        plt.show()

if __name__ == '__main__':
    params = {
        'beta': 0,
        'L': 100,
        'frames': 1000,
        'save_result': False,
        'plot_raw_result': True,
    }
    main = CuttingProfile(**params)
    main.start()
