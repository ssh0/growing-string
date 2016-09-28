#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-08-19


from eden import Eden
from Optimize import Optimize_powerlaw
from surface import get_surface_points, set_labels, get_labeled_position
from roughness import *
import matplotlib.pyplot as plt
import numpy as np


class Roughness_Eden(Eden):
    def __init__(self, L=60, frames=1000):
        Eden.__init__(self, Lx=L, Ly=L, plot=False, frames=frames)


if __name__ == '__main__':

    fig, ax = plt.subplots()
    for i in range(1):
        main = Roughness_Eden(L=60, frames=1000)
        main.execute()
        # main = Roughness(L=120, frames=3000)

        # 隣接格子点に同じラベルを振る
        # 元
        # (theta, r), R_t, label_list = eval_fluctuation_on_surface(main, np.array(main.points), test=True)
        # index_sorted = np.argsort(theta)
        # theta, r, label_list = theta[index_sorted], r[index_sorted], label_list[index_sorted]
        # plot_to_veirfy(theta, r, R_t, label_list)
        # plt.show()

        # 最大クラスターのみ表示
        (theta, r), R_t, label_list = eval_fluctuation_on_surface(main, np.array(main.points))
        index_sorted = np.argsort(theta)
        theta = theta[index_sorted]
        r = r[index_sorted]
        label_list = label_list[index_sorted]

        # plot_to_veirfy(theta, r, R_t, label_list)
        # plt.show()

        res_width, res_std = eval_std_various_width(theta, r, R_t)
        ax = plot_result(res_width, res_std, ax)

        # FIXME: フィッティング領域の選択の自動化
        fitting(ax, res_width, res_std, 7, 38)  # <- {L: 60, frames=1000}
        # fitting(ax, res_width, res_std, 5, 32)  # <- {L: 120, frames=3000}
        ax.set_title('Roughness (averaged) at some width on Eden model')

    plt.show()

