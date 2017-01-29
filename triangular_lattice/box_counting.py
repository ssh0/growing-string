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
# from optimize import Optimize_powerlaw
# from optimize import Optimize_linear
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import save_data
# import save_meta


class BoxCounting(object):
    def __init__(self, frames, beta,
                 frames_list, N_L,
                 save_fitting=False,
                 save_fitting_dir=''):
        L = (frames + 2) * 2
        self.L = L
        self.pbar = tqdm(total=frames)
        self.frames = frames
        self.frames_list = frames_list
        self.beta = beta
        self.N_L = N_L
        self.t = 1
        self.save_fitting = save_fitting
        self.save_fitting_dir = save_fitting_dir

    def start(self):
        self.get_cutting_sizes(self.N_L)
        self.main = Main(
            Lx=self.L,
            Ly=self.L,
            size=[3,],
            plot=False,
            frames=self.frames,
            strings=[{'id': 1, 'x': self.L/4, 'y': self.L/2, 'vec': [0, 4]}],
            beta=self.beta,
            post_function=self.calc_fractal_dim
        )
        self.pbar.close()

    def calc_fractal_dim(self, main, i, s):
        if len(s.vec) + 1 - 3 not in self.frames_list:
            self.pbar.update(1)
            self.t += 1
            return None

        self.x = main.lattice_X[np.array(s.pos.T).tolist()]
        self.y = main.lattice_Y[np.array(s.pos.T).tolist()]

        N = np.array(self.get_results_each_subclusters())

        # index_end = len(np.where(N == 1)[0])

#         if index_end > len(N) - 2:
#             self.pbar.update(1)
#             self.t += 1
#             return 0.

#         if index_end == 0 or index_end == 1:
#             index_end = len(N)
#         else:
#             index_end = -(index_end - 1)

#         # optimizer = Optimize_powerlaw(
#         #     args=(self.cutting_size_xs[:index_end], N[:index_end]),
#         #     parameters=[100., -1.5])
#         optimizer = Optimize_linear(
#             args=(np.log(self.cutting_size_xs[:index_end]), np.log(N[:index_end])),
#             parameters=[-1.5, np.log(100)])
#         result = optimizer.fitting()
#         # print result

#         if self.save_fitting:
#             if self.t in np.linspace(1, self.frames, num=10, dtype=np.int):
#                 fig, ax = plt.subplots()
#                 ax.loglog(self.cutting_size_xs, N, 'o')
#                 # ax.loglog(self.cutting_size_xs[:index_end],
#                 #         optimizer.fitted(self.cutting_size_xs[:index_end]),
#                 #         '-', label='D = %f' % result['D'])
#                 ax.loglog(self.cutting_size_xs[:index_end],
#                         np.exp(optimizer.fitted(np.log(self.cutting_size_xs[:index_end]))),
#                         '-', label='D = %f' % result['a'])
#                 ax.legend(loc='best')
#                 ax.set_title('Coarse graining (frames = {})'.format(self.t))
#                 ax.set_xlabel(r'$\varepsilon$')
#                 ax.set_ylabel(r'$N(\varepsilon)$')
#                 # filename = "results/img/box_counting/2016-11-25/"
#                 filename = self.save_fitting_dir
#                 filename += "beta=%2.2f_fitting_frame=%d.png" % (self.beta, self.t)
#                 fig.savefig(filename)
#                 plt.close(fig)
        # return result['D']
        # return result['a']

        self.pbar.update(1)
        self.t += 1
        return np.array([self.cutting_size_xs, N]).T

    def get_cutting_sizes(self, N_L):
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
                int,
                np.logspace(0, np.log(self.L) / np.log(1.5), base=1.5, num=N_L)
            ))))
        # self.cutting_size_ys = self.cutting_size_xs * (np.sqrt(3) / 2)
        self.cutting_size_ys = self.cutting_size_xs
        self.cutting_size_max_width = self.cutting_size_xs[-1]
        self.cutting_size_max_height = self.cutting_size_ys[-1]

        self.cutting_sizes = np.array([self.cutting_size_xs,
                                       self.cutting_size_ys]).T
        return

    def diecutting_one_cluster(self, width, height, x0, y0):
        return np.any((self.x > x0) & 
                      (self.y > y0) &
                      (self.x < (x0 + width)) &
                      (self.y < (y0 + height))
                      )

    def get_results_each_subclusters(self):
        res = []
        for width, height in self.cutting_sizes:
            N = int(self.L / width)
            x0s = np.arange(N + 1) * width - 0.25
            y0s = np.arange(N + 1) * height - (np.sqrt(3) / 4)
            _res = 0
            for x0 in x0s:
                for y0 in y0s:
                    if self.diecutting_one_cluster(width, height, x0=x0, y0=y0):
                        _res += 1
            res.append(_res)
        return res


# def main(beta, plot):
#     print "beta = %2.2f" % beta

#     frames = 1000
#     box_counting = BoxCounting(frames=frames, beta=beta,
#                                save_fitting=True,
#                                save_fitting_dir="results/img/box_counting/2016-12-01/")
#     box_counting.start()
#     Ds = -np.array(box_counting.main.pre_func_res)
#     T = np.arange(frames)

#     # base = "results/data/box_counting/beta=%2.2f_" % beta
#     # base = "results/data/box_counting/modified/beta=%2.2f_" % beta
#     # base = "results/data/box_counting/2016-11-19/beta=%2.2f_" % beta  # εの刻みを多くしたバージョン
#     # base = "results/data/box_counting/2016-11-25/beta=%2.2f_" % beta  # クラスターがシミュレーションじのシステムサイズ内に入るようにしたもの
#     base = "results/data/box_counting/2016-12-01/beta=%2.2f_" % beta  # + optimizeのアルゴリズムを変更
#     save_data.save(base, beta=beta, L=box_counting.L,
#                    frames=box_counting.frames, Ds=Ds)
#     save_meta.save(base, beta=beta, L=box_counting.L,
#                    frames=box_counting.frames)

#     if plot:
#         fig, ax = plt.subplots()

#         ax.plot(np.sqrt(T), Ds, '.')
#         # ax.legend(loc='best')
#         ax.set_title('Fractal dimension')
#         ax.set_xlabel(r'$T$')
#         ax.set_ylabel(r'$D(T)$')
#         plt.show()


def box_count(beta, frames_list, N_L=20, num_of_strings=100):
    frames = np.max(frames_list)

    string_num = 1
    print 'string ({}/{})'.format(string_num, num_of_strings)
    bc = BoxCounting(frames=frames, beta=beta,
                     frames_list=frames_list,
                     N_L=N_L,
                     # save_fitting=True,
                     # save_fitting_dir="results/img/box_counting/2016-12-01/")
                     )
    bc.start()
    _N = np.array([n for n in bc.main.post_func_res if n is not None])
    Ns = {frames_list[i]: _N[i] for i in range(len(frames_list))}

    for s in range(num_of_strings - 1):
        string_num += 1
        print 'string ({}/{})'.format(string_num, num_of_strings)
        bc = BoxCounting(frames=frames, beta=beta,
                        frames_list=frames_list,
                        N_L=N_L,
                        # save_fitting=True,
                        # save_fitting_dir="results/img/box_counting/2016-12-01/")
                        )
        bc.start()
        _N = np.array([n for n in bc.main.post_func_res if n is not None])
        for i, frames in enumerate(frames_list):
            Ns[frames] = np.vstack((Ns[frames], _N[i]))

    for frames in frames_list:
        Ls, N = Ns[frames].T
        sorted_index = np.argsort(Ls)
        Ls, N = Ls[sorted_index], N[sorted_index]
        # save_data.save("./results/data/box_counting/2017-01-27/" +
        save_data.save("./results/data/box_counting/2017-01-29/" +
                       "beta=%2.2f_frames=%d_" % (beta, frames),
                       num_of_strings=num_of_strings,
                       N_L=N_L, beta=beta, L=bc.L, frames=frames, Ls=Ls, N=N)


if __name__ == '__main__':
    num_of_strings = 30

    parser = argparse.ArgumentParser()
    parser.add_argument('beta', type=float, nargs=1,
                        help='parameter beta (inverse temparature)')
    args = parser.parse_args()
    beta = args.beta[0]
    print "beta = %2.2f" % beta

    # frames_list = np.linspace(200, 600, num=3, dtype=np.int)
    frames_list = np.linspace(200, 2000, num=10, dtype=np.int)
    box_count(beta, frames_list, N_L=20, num_of_strings=num_of_strings)
