#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-10-05


import time
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from growing_string import Main
import matplotlib.pyplot as plt


max_dists = []
sample_num = 10
betas_num = 10
betas = np.linspace(0., 10., num=betas_num)
frames = 400
L = 1000

params = {
    'Lx': L,
    'Ly': L,
    'frames': frames,
    'size': [3,] * 1,
    'plot': False,
    'save_image': False,
    'strings': [{'id': 1, 'x': L/4, 'y': L/2, 'vec': [0, 4]}],
}

def calc_max_radius(beta):
    return max([_calc_max_radius(beta) for i in range(sample_num)])

def _calc_max_radius(beta):
    main = Main(beta=beta, **params)
    s = main.strings[0]
    N = float(len(s.vec) + 1)
    pos = list(s.pos.T)
    x = main.lattice_X[pos]
    y = main.lattice_Y[pos]
    X = main.lattice_X[L / 4, L / 2]
    Y = main.lattice_Y[L / 4, L / 2]
    r = np.sqrt((x - X) ** 2 + (y - Y) ** 2)
    return np.max(r)

pool = Pool(6)
ite = pool.imap(calc_max_radius, betas)
for ret in tqdm(ite, total=betas_num):
    max_dists.append(ret)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(betas, max_dists)
ax.set_title("Max distance from the starting point")
ax.set_xlim(0, max(betas) + 1)
ax.set_ylim(min(max_dists) - 0.1, max(1., max(max_dists) + 0.1))
ax.set_xlabel(r'$\beta$')
ax.set_ylabel(r'$R$')
fn = "./results/img/max_radius/"
fn += "frames=%d" % frames + "_sample=%d" % sample_num
fn += time.strftime("_%y%m%d_%H%M%S")
fn += ".png"
fig.savefig(fn)
plt.close()

