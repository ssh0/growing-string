#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-10-03


import time
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from radius import *
import matplotlib.pyplot as plt


# # Create result images
# betas_num = 10
# betas = np.linspace(0., 10., num=betas_num)
# frames = 500
# Ls = [500, 1000]
# test_data = [(beta, L) for beta in betas for L in Ls]

# for beta, L in tqdm(test_data):
#     main(beta, L=L, frames=frames, plot=False, 
#          output="./results/img/radius/beta=" + str(beta) +
#          "_L=%03d" % L +
#          "_frames=" + str(frames) +
#          "_fitted.png",
#          save_image=True,
#          filename_image="./results/img/radius/beta=" + str(beta) +
#          "_L=%03d" % L +
#          "_frames=" + str(frames) +
#          "_raw.png"
#          )


Rs = []
err = []
sample_num = 100
betas_num = 20
betas = np.linspace(0., 10., num=betas_num)
frames = 500
L = 1000

def calc_radius(beta):
    res = np.array([main(beta, L=L, frames=frames, plot=False,
                         plot_optimized=False)
                    for i in range(sample_num)])
    return {'average': np.average(res),
            'var': np.var(res) }

pool = Pool(7)
ite = pool.imap(calc_radius, betas)
for ret in tqdm(ite, total=betas_num):
    Rs.append(ret['average'])
    err.append(ret['var'])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.errorbar(betas, Rs, err)
ax.set_title("Radius of rotation")
ax.set_xlim(0, max(betas) + 1)
ax.set_ylim(min(Rs) - 0.1, max(1., max(Rs) + 0.1))
ax.set_xlabel(r'$\beta$')
ax.set_ylabel(r'$D$')
fig.savefig("./results/img/radius/fitted_result_sample=%d" % sample_num + ".png")
plt.close()

