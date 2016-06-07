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
# data = np.load("2016-06-02.npz")
# data = np.load("2016-06-03_80.npz")
# data = np.load("2016-06-03_120.npz")
data = np.load("2016-06-07_40.npz")
T = data['T']
sizeset = data['sizeset']

df = pd.DataFrame(T, index=sizeset)
# df = pd.DataFrame(1./T, index=sizeset)
# print df.T
df = df.T

# ===================
# # box = df.plot.box(positions=sizeset)
# box = df.plot.box(positions=sizeset, showfliers=False, logx=True)

# box.set_title("Deadlock time for each string size")
# box.set_xlabel("String size $N$")
# # box.set_ylabel("Deadlock time $T$")
# box.set_ylabel("Inverse of deadlock time $1/T$")
# ===================

# ===================
fig, ax = plt.subplots()

# sizeset =
# [  8   9  10  11  12  13  14  15  16  17  18  20  21  23  24  26  28  30
#   32  35  37  40  43  46  50  54  57  62  66  71  76  82  88  95 102 109
#  117 126 135 145 156 167 179 192 207 222 238 256]
n = 256
if not n in sizeset:
    raise UserWarning("n is not in sizeset")

df[n].plot.hist(ax=ax, bins=100, normed=True, logy=True)
ax.set_title("Deadlock time distribution for string size $N = {}$".format(n))

# for n in [8, 30, 256]:
#     df[n].plot.hist(ax=ax, bins=50, normed=True, logy=True)
# ax.set_title("Deadlock time distribution for string size $N$")

# Lx=80のときと比較 ================
# data2 = np.load("2016-06-03_80.npz")
# T2 = data2['T']
# sizeset2 = data2['sizeset']
# df2 = pd.DataFrame(T2, index=sizeset2)
# df2 = df2.T
# df[n].plot.hist(ax=ax, bins=150, logy=True, ylim=(0, 1000), alpha=0.6)
# df2[n].plot.hist(ax=ax, bins=150, logy=True, ylim=(0, 1000), alpha=0.6)
# ==================================

# Lx=120のときと比較 ================
# data3 = np.load("2016-06-03_120.npz")
# T3 = data3['T']
# sizeset3 = data3['sizeset']
# df3 = pd.DataFrame(T3, index=sizeset3)
# df3 = df3.T
# df[n].plot.hist(ax=ax, bins=150, logy=True, ylim=(0, 1000), alpha=0.6)
# df3[n].plot.hist(ax=ax, bins=150, logy=True, ylim=(0, 1000), alpha=0.6)
# ==================================

ax.set_xlabel("Deadlock time $T$")
# ===================

plt.show()
