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
import save_data


beta = 8.
num_of_strings = 100
frames = 500
L = 1000

current_time = time.strftime("%y%m%d_%H%M%S")

res = []
for i in tqdm(range(num_of_strings)):
    res.append(main(beta, L=L, frames=frames, plot=False,
                    plot_optimized=False))
res = np.array(res)

save_data.save("results/data/radius/beta=%2.2f_" % beta,
               num_of_strings=num_of_strings,
               beta=beta, L=L, frames=frames, D=res)

