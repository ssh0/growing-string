#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-11-16
"""ボックスカウンティング法によってクラスターのフラクタル次元の時間変化を求める。
最終的にはフラクタル次元の時間変化がtanh(t)に従うことを示したい。
~~シミュレーション時に用意する格子サイズは，2の階乗にし，分割が行い易くなるようにする。~~
格子サイズは，フィッティング時のサンプル数を確保するために，上のようなことを考える必要はない
"""

from diecutting import DieCutting
from growing_string import Main
from optimize import Optimize_powerlaw
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import save_data
import save_meta


class BoxCounting(object):
    def __init__(self, frames, beta, c=0.4, L_power=10):
        L = 2 ** L_power
        if frames > (L*L) * 0.9:
            raise ValueError("`frames` must be smaller than 0.9 times `L` * `L`.")
        self.L = L
        self.pbar = tqdm(total=frames)
        self.frames = frames
        self.beta = beta
        self.weight_const = c
        self.t = 1

    def start(self):
        self.get_cutting_sizes()
        self.main = Main(
            Lx=self.L,
            Ly=self.L,
            size=[3,],
            plot=False,
            frames=self.frames,
            strings=[{'id': 1, 'x': self.L/4, 'y': self.L/2, 'vec': [0, 4]}],
            beta=self.beta,
            weight_const=self.weight_const,
            pre_function=self.calc_fractal_dim
        )

    def calc_fractal_dim(self, main, i, s):

        self.x = main.lattice_X[np.array(main.strings[0].pos.T).tolist()]
        self.y = main.lattice_Y[np.array(main.strings[0].pos.T).tolist()]

        N = np.array(self.get_results_each_subclusters())
        index_end = len(np.where(N == 1)[0])

        if index_end > len(N) - 2:
            self.pbar.update(1)
            self.t += 1
            return 0.

        if index_end == 0 or index_end == 1:
            index_end = len(N)
        else:
            index_end = -(index_end - 1)

        optimizer = Optimize_powerlaw(
            args=(self.cutting_size_xs[:index_end], N[:index_end]),
            parameters=[0., -1.5])
        result = optimizer.fitting()

        if self.t in np.linspace(1, self.frames, num=10, dtype=np.int):
            fig, ax = plt.subplots()
            ax.loglog(self.cutting_size_xs, N, 'o')
            ax.loglog(self.cutting_size_xs[:index_end],
                      optimizer.fitted(self.cutting_size_xs[:index_end]),
                      '-', label='D = %f' % result['D'])
            ax.legend(loc='best')
            ax.set_title('Coarse graining (frames = {})'.format(self.t))
            ax.set_xlabel(r'$\varepsilon$')
            ax.set_ylabel(r'$N(\varepsilon)$')
            filename = "results/img/box_counting/2016-11-25/"
            filename += "beta=%2.2f_fitting_frame=%d.png" % (self.beta, self.t)
            fig.savefig(filename)

        self.pbar.update(1)
        self.t += 1

        return result['D']

    def get_cutting_sizes(self):
        """Create the cutting size list for simulation

        self.X0: x coordinates of the left buttom corner
        self.Y0: y coordinates of the left buttom corner
        self.cutting_size_max_width: max width of the cutting size
        self.cutting_size_max_height: max height of the cutting size
        self.cutting_size_xs: cutting size list
        self.cutting_size_ys: cutting size list
        self.cutting_sizes: ndarray [[cutting_size_xs[0], cutting_size_ys[0]],
                                     [cutting_size_xs[1], cutting_size_ys[1]],
                                         ...
                                    ]

        In this funciton, cutting size is determined by which the whole region
        is in the cluster.
        """

        self.X0 = 0.
        self.Y0 = 0.

        # self.cutting_size_xs = np.array([2 ** (i + 1) for i
        #                                  in range(int(np.log2(self.L)))])
        # self.cutting_size_xs = np.arange(1, self.L / 2)
        self.cutting_size_xs = np.array(
            sorted(set(map(
                int, np.logspace(0, np.log(self.L) / np.log(1.5), base=1.5)
            ))))
        self.cutting_size_ys = self.cutting_size_xs * (np.sqrt(3) / 2)
        self.cutting_size_max_width = self.cutting_size_xs[-1]
        self.cutting_size_max_height = self.cutting_size_ys[-1]

        self.cutting_sizes = np.array([self.cutting_size_xs,
                                       self.cutting_size_ys]).T
        return

    def diecutting_one_cluster(self, width, height, x0, y0):
        # return np.any(self.x > x0)         and \
        #     np.any(self.y > y0)            and \
        #     np.any(self.x < (x0 + width))  and \
        #     np.any(self.y < (y0 + height))
        return np.any(
            np.logical_and(
                np.logical_and(self.x > x0, self.x < (x0 + width)),
                np.logical_and(self.y > y0, self.y < (y0 + height)),
            )
        )

    def get_results_each_subclusters(self):
        res = []
        for width, height in self.cutting_sizes:
            N = int(self.L / width)
            x0s = np.arange(N) * width - 0.25
            y0s = np.arange(N) * height - (np.sqrt(3) / 4)
            _res = 0
            for x0 in x0s:
                for y0 in y0s:
                    if self.diecutting_one_cluster(width, height, x0=x0, y0=y0):
                        _res += 1
            res.append(_res)
        return res


def main(beta, plot):
    print "beta = %2.2f" % beta

    frames = 500
    box_counting = BoxCounting(frames=frames, beta=beta, L_power=10)
    box_counting.start()
    Ds = -np.array(box_counting.main.pre_func_res)
    T = np.arange(frames)

    # base = "results/data/box_counting/beta=%2.2f_" % beta
    # base = "results/data/box_counting/modified/beta=%2.2f_" % beta
    # base = "results/data/box_counting/2016-11-19/beta=%2.2f_" % beta  # εの刻みを多くしたバージョン
    base = "results/data/box_counting/2016-11-25/beta=%2.2f_" % beta  # クラスターがシミュレーションじのシステムサイズ内に入るようにしたもの
    save_data.save(base, beta=beta, L=box_counting.L,
                   frames=box_counting.frames, Ds=Ds)
    save_meta.save(base, beta=beta, L=box_counting.L,
                   frames=box_counting.frames)

    if plot:
        fig, ax = plt.subplots()

        ax.plot(np.sqrt(T), Ds, '.')
        # ax.legend(loc='best')
        ax.set_title('Fractal dimension')
        ax.set_xlabel(r'$T$')
        ax.set_ylabel(r'$D(T)$')
        plt.show()


if __name__ == '__main__':
    beta = 10.
    main(beta, plot=False)

