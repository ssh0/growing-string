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

result_data_path = "results/data/radius/beta=%2.2f" % beta
result_data_path += "_" + current_time
result_data_path += ".npz"
np.savez(result_data_path,
         num_of_strings=num_of_strings,
         beta=beta,
         L=L,
         frames=frames,
         D=res)

