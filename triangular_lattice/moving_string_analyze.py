#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-06-01


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load data ===================================================================
# data = np.load("2016-05-31.npz")
data = np.load("2016-06-02.npz")

# df = pd.DataFrame(data['T'], index=data['sizeset'])
df = pd.DataFrame(1./data['T'], index=data['sizeset'])
# print df.T
df = df.T

# box = df.plot.box(positions=data['sizeset'])
box = df.plot.box(positions=data['sizeset'], showfliers=False, logx=True)

box.set_title("Deadlock time for each string size")
box.set_xlabel("String size $N$")
# box.set_ylabel("Deadlock time $T$")
box.set_ylabel("Inverse of deadlock time $1/T$")
plt.show()
