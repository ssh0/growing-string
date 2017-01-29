#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2017-01-21

import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from growing_string import Main
from optimize import Optimize_powerlaw
import save_data


def mass_for_beta_one(beta, frames_list, N_r=100, num_of_strings=100):
    frames = np.max(frames_list)
    center_sample = int(np.min(frames_list) / 2)
    L = (frames + 1) * 2

    def calc_mass_in_r(self, i, s):
        N = len(s.vec) + 1
        if N - 3 not in frames_list:
            return None

        pos = list(s.pos.T)
        x, y = self.lattice_X[pos], self.lattice_Y[pos]
        X, Y = np.average(x), np.average(y)
        R = np.sqrt(np.sum((x - X) ** 2 + (y - Y) ** 2) / float(N))
        dist = np.sqrt((x - X) ** 2 + (y - Y) ** 2)
        r = np.logspace(1, np.log2(max(dist)), num=N_r, base=2.)
        centers_index = sorted(random.sample(range(N), center_sample))
        M = []
        for _r in r:
            res = []
            for c in centers_index:
                index_x, index_y = s.pos[c]
                dist = np.sqrt((x - self.lattice_X[index_x, index_y]) ** 2
                                + (y - self.lattice_Y[index_x, index_y]) ** 2)
                res.append(len(np.where(dist < _r)[0]))
            M.append(np.average(res))
        return np.array([r, M]).T

    main = Main(Lx=L, Ly=L, plot=False,
                frames=frames,
                beta=beta,
                strings=[{'id': 1, 'x': L/4, 'y': L/2, 'vec': [0, 4]}],
                post_function=calc_mass_in_r)
    _M = np.array([m for m in main.post_func_res if m is not None])
    Ms = {frames_list[i]: _M[i] for i in range(len(frames_list))}

    for s in tqdm(range(num_of_strings - 1)):
        main = Main(Lx=L, Ly=L, plot=False,
                    frames=frames,
                    beta=beta,
                    strings=[{'id': 1, 'x': L/4, 'y': L/2, 'vec': [0, 4]}],
                    post_function=calc_mass_in_r)
        _M = np.array([m for m in main.post_func_res if m is not None])
        # print _M.shape
        for i, frames in enumerate(frames_list):
            Ms[frames] = np.vstack((Ms[frames], _M[i]))

    for frames in frames_list:
        r, M = Ms[frames].T
        sorted_index = np.argsort(r)
        r, M = r[sorted_index], M[sorted_index]
        save_data.save("./results/data/mass_in_r/beta=%2.2f_frames=%d_" % (beta, frames),
                       num_of_strings=num_of_strings,
                       N_r=N_r, beta=beta, L=L, frames=frames, r=r, M=M)


if __name__ == '__main__':

    # frames_list = np.linspace(200, 600, num=3, dtype=np.int)
    frames_list = np.linspace(200, 2000, num=10, dtype=np.int)

    parser = argparse.ArgumentParser()
    parser.add_argument('beta', type=float, nargs=1,
                        help='parameter beta')
    args = parser.parse_args()
    beta = args.beta[0]
    # beta = 0.

    # mass_for_beta_one(beta, frames_list, N_r=4, num_of_strings=3)
    mass_for_beta_one(beta, frames_list, N_r=100, num_of_strings=100)
