#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-10-18

import time
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import matplotlib.pyplot as plt

Ds = []
err = []
betas = [0., 2., 4., 6., 8.]

result_data_paths = [
    './results/data/radius/beta=0.00_161018_235815.npz',
    './results/data/radius/beta=2.00_161018_235822.npz',
    './results/data/radius/beta=4.00_161018_235827.npz',
    './results/data/radius/beta=6.00_161018_235834.npz',
    './results/data/radius/beta=8.00_161018_235842.npz',
]

for result_data_path in result_data_paths:
    data = np.load(result_data_path)
    beta = data['beta']
    num_of_strings = data['num_of_strings']
    L = data['L']
    frames = data['frames']
    D = data['D']
    print D
    Ds.append(np.average(D))
    err.append(np.var(D))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.errorbar(betas, Ds, err)
ax.set_title("Fractal dimension by the relationship between steps and the radius of rotation")
ax.set_xlim(0, max(betas) + 1)
ax.set_ylim(min(Ds) - 0.1, max(1., max(Ds) + 0.1))
ax.set_xlabel(r'$\beta$')
ax.set_ylabel(r'$D$')
fig.savefig("./results/img/radius/fractal_dim_sample=%d" % num_of_strings +
            "_" + time.strftime("%y%m%d_%H%M%S") + ".png")
plt.close()

