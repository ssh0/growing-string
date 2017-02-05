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
import save_data as sd


class CuttingProfile(object):
    def __init__(self, frames, beta, c=0.5, L=100, save_result=False,
                 plot_raw_result=False, _plot_dist_to_verify=False):
        self.L = L
        # self.pbar = tqdm(total=frames)
        self.frames = frames
        self.beta = beta
        self.weight_const = c
        self.t = 1
        self.save_result = save_result
        self.plot_raw_result = plot_raw_result
        self._plot_dist_to_verify = _plot_dist_to_verify

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
        self.relative_positions = self.get_relative_positions()

        if self.save_result:
            sd.save("results/data/cutting_profile/" +
                    "frames=%d_beta=%2.2f_" % (self.frames, self.beta),
                    beta=self.beta, L=self.L, frames=self.frames,
                    weight_const=self.weight_const,
                    cutting_profiles=self.cutting_profiles
                    )
        if self.plot_raw_result:
            self.plot_result()

        if self._plot_dist_to_verify:
            self.plot_dist_to_verify()

    def get_cutting_profiles(self):
        """6つの方向でそれぞれに対して，ベクトルの向きに対して左側に
        閉じている領域要素の位置(格子座標)を記録
        """
        return [self.get_cutting_profile_for(k) for k in range(6)]

    def get_cutting_profile_for(self, k):
        """k方向のベクトルに対して左側に閉じた部分領域の位置(格子座標)を取得"""
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

    def get_relative_positions(self):
        """重心からの相対座標を求める"""
        s = self.main.strings[0]
        lX = self.main.lattice_X
        lY = self.main.lattice_Y
        self.x0 = np.average(lX[s.pos_x, s.pos_y])
        self.y0 = np.average(lY[s.pos_x, s.pos_y])
        self.real_position = {}
        self.real_position_x = {}
        self.real_position_y = {}
        relative_positions = {}
        for k in range(6):
            positions = np.array(self.cutting_profiles[k]).T
            if len(positions) == 0:
                continue
            index_x, index_y = positions[0], positions[1]
            # self.real_position.append(
            #     np.array(lX[index_x, index_y], lY[index_x, index_y]).T)
            X = lX[index_x, index_y]
            Y = lY[index_x, index_y]
            self.real_position_x[k] = X
            self.real_position_y[k] = Y
            self.real_position[k] = np.array([X, Y]).T
            relative_positions[k] = np.array([X - self.x0, Y - self.y0]).T
        return relative_positions

    def plot_result(self):
        self.main.plot_all()
        lX = self.main.lattice_X
        lY = self.main.lattice_Y
        self.main.ax.plot(self.x0, self.y0, '.', color='k')
        for k in range(6):
            positions = np.array(self.cutting_profiles[k]).T
            if len(positions) == 0:
                continue
            index_x, index_y = positions[0], positions[1]
            self.main.ax.plot(lX[index_x, index_y], lY[index_x, index_y], '.',
                              label='vec: {}'.format(k),
                              color=cm.rainbow(float(k) / 5))

        self.main.ax.legend(loc='best')
        plt.show()

    def plot_dist_to_verify(self):
        fig, ax = plt.subplots()
        for k in range(6):
            if self.relative_positions.has_key(k):
                ax.plot(*self.relative_positions[k].T, marker='.', ls='none',
                        label='vec: {}'.format(k),
                        color=cm.rainbow(float(k) / 5))
        ax.legend(loc='best')
        plt.show()

if __name__ == '__main__':
    params = {
        'beta': 3.,
        'L': 120,
        'frames': 1000,
        'save_result': False,
        'plot_raw_result': True,
        '_plot_dist_to_verify': True,
    }
    main = CuttingProfile(**params)
    main.start()
