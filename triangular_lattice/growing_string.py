#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-07-12


from __future__ import print_function
from triangular import LatticeTriangular as LT
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.animation as animation
from base import Main as base
from strings import String
import numpy as np
import operator
import pprint

pp = pprint.PrettyPrinter(indent=4)


class Main(base):
    """任意に設定したstringの近傍点に点を追加し成長させるモデル

    グラフ上で左端と右端に固定されたstringの近傍点を探索，ランダム(後には曲げ
    弾性による重み付けの効果を追加)に選択し，stringを成長させていくモデル
    """

    def __init__(self, Lx=40, Ly=40,
                 boundary={'h': 'periodic', 'v': 'periodic'},
                 size=[5, 4, 10, 12],
                 plot=True,
                 plot_surface=True,
                 save_image=False,
                 save_video=False,
                 filename_image="",
                 filename_video="",
                 frames=1000,
                 beta = 2.,
                 interval=1,
                 weight_const=0.5,
                 strings=None,
                 pre_function=None,
                 post_function=None):
        """Init function of Main class.

        Lx (int (even)): 格子のx方向(グラフではy軸)の格子点数
        Ly (int (even)): 格子のy方向(グラフではx軸)の格子点数
        """
        # Create triangular lattice with given parameters
        # self.lattice = LT(np.zeros((Lx, Ly), dtype=np.int),
        #                   scale=float(max(Lx, Ly)), boundary=boundary)
        self.lattice = LT(
            np.zeros((Lx, Ly), dtype=np.int),
            scale=float(max(Lx, Ly)),
            boundary=boundary
        )
        self.lattice_X = self.lattice.coordinates_x.reshape(
            self.lattice.Lx,
            self.lattice.Ly
        )
        self.lattice_Y = self.lattice.coordinates_y.reshape(
            self.lattice.Lx,
            self.lattice.Ly
        )
        self.occupied = np.zeros((Lx, Ly), dtype=np.bool)
        self.number_of_lines = Lx

        if strings is None:
            # Put the strings to the lattice
            self.strings = self.create_random_strings(len(size), size)
        else:
            self.strings = [String(self.lattice, **st) for st in strings]
        for string in self.strings:
            self.occupied[string.pos_x, string.pos_y] = True


        self.plot = plot
        self.plot_surface = plot_surface
        self.save_image = save_image
        self.save_video = save_video
        if self.save_image:
            if filename_image == "":
                raise AttributeError("`filename_image` is empty.")
            else:
                self.filename_image = filename_image

        if self.save_video:
            if self.plot:
                raise AttributeError("`save` and `plot` method can't be set both True.")
            if filename_video == "":
                raise AttributeError("`filename_video` is empty.")
            else:
                self.filename_video = filename_video


        self.interval = interval
        self.frames = frames

        # 逆温度
        self.beta = beta
        # self.beta = 100. # まっすぐ(≒低温極限)
        # self.beta = 10. # まっすぐ
        # self.beta = 0. # 高温極限
        # self.beta = 5. # 中間的
        self.weight_const = weight_const

        self.bonding_pairs = {i: {} for i in range(len(self.strings))}
        for key in self.bonding_pairs.keys():
            value = self.get_bonding_pairs(
                s=self.strings[key],
                indexes=[[0, len(self.strings[key].pos)]]
            )

            # TODO: 隣接点がないとき，全体のシミュレーションを終了する
            # if len(value) == 0:
            #     return False

            self.bonding_pairs[key] = value

        # pp.pprint(self.bonding_pairs)
        # print(self.strings[0].pos)
        # pp.pprint(self.bonding_pairs[0])


        # pp.pprint(self.bonding_pairs)
        # return None

        self.pre_function = pre_function
        self.post_function = post_function
        self.pre_func_res = []
        self.post_func_res = []

        # Plot triangular-lattice points, string on it, and so on
        if self.plot:
            self.plot_all()
            self.start_animation()
        elif self.save_video:
            self.plot_all()
            self.start_animation(filename=self.filename_video)
        else:
            t = 0
            while t < self.frames:
                try:
                    self.update()
                    t += 1
                except StopIteration:
                    break

        if self.save_image:
            if not self.__dict__.has_key('fig'):
                self.plot_all()
            self.fig.savefig(self.filename_image)
            plt.close()
            # print("Image file is successfully saved at '%s'." % filename_image)

    def __update_dict(self, dict, key, value):
        if dict.has_key(key):
            dict[key].append(value)
        else:
            dict[key] = [value]

    def dot(self, v, w):
        """0〜5で表された6つのベクトルの内積を計算する。

        v, w (int): ベクトル(0〜5の整数で表す)"""
        if (w + 6 - v) % 6 == 0:
            return 1
        elif (w + 6 - v) % 6 == 1 or (w + 6 - v) % 6 == 5:
            return 0.5
        elif (w + 6 - v) % 6 == 2 or (w + 6 - v) % 6 == 4:
            return -0.5
        elif (w + 6 - v) % 6 == 3:
            return -1.

    def plot_all(self):
        """軸の設定，三角格子の描画，線分描画要素の用意などを行う

        ここからFuncAnimationを使ってアニメーション表示を行うようにする
        """
        self.fig, self.ax = plt.subplots(figsize=(8, 8))

        self.lattice_X = self.lattice.coordinates_x
        self.lattice_Y = self.lattice.coordinates_y
        X_min, X_max = min(self.lattice_X) - 0.1, max(self.lattice_X) + 0.1
        Y_min, Y_max = min(self.lattice_Y) - 0.1, max(self.lattice_Y) + 0.1
        self.ax.set_xlim([X_min, X_max])
        self.ax.set_ylim([Y_min, Y_max])
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.set_aspect('equal')

        triang = tri.Triangulation(self.lattice_X, self.lattice_Y)
        self.ax.triplot(triang, color='#d5d5d5', marker='.', markersize=1)

        self.lines = [self.ax.plot([], [], linestyle='-',
                                   color='black',
                                   markerfacecolor='black',
                                   markeredgecolor='black')[0]
                      for i in range(self.number_of_lines)]
        if self.plot_surface:
            self.lines.append(self.ax.plot([], [], '.', color='#ff0000')[0])

        self.lattice_X = self.lattice_X.reshape(self.lattice.Lx,
                                                self.lattice.Ly)
        self.lattice_Y = self.lattice_Y.reshape(self.lattice.Lx,
                                                self.lattice.Ly)
        self.plot_string()

    def start_animation(self, filename=""):
        if self.__dict__.has_key('frames'):
            frames = self.frames
        else:
            frames = 1000

        def init_func(*arg):
            return self.lines

        ani = animation.FuncAnimation(self.fig, self.update, frames=frames,
                                      init_func=init_func,
                                      interval=self.interval,
                                      blit=True, repeat=False)
        if filename != "":
            try:
                ani.save(filename, codec="libx264", bitrate=-1, fps=30)
            except:
                print("Can't saved.")
            else:
                print("Animation is successfully saved at '%s'." % filename)
        else:
            plt.show()

    def plot_string(self):
        """self.strings内に格納されているStringを参照し，グラフ上に図示する
        """
        # print self.string.pos, self.string.vec
        i = 0  # to count how many line2D object
        for s in self.strings:
            start = 0
            for j, pos1, pos2 in zip(range(len(s.pos) - 1), s.pos[:-1], s.pos[1:]):
                dist_x = abs(self.lattice_X[pos1[0], pos1[1]] -
                            self.lattice_X[pos2[0], pos2[1]])
                dist_y = abs(self.lattice_Y[pos1[0], pos1[1]] -
                            self.lattice_Y[pos2[0], pos2[1]])
                # print j, pos1, pos2
                # print dist_x, dist_y
                if dist_x > 1.5 * self.lattice.dx or dist_y > 1.5 * self.lattice.dy:
                    x = s.pos_x[start:j + 1]
                    y = s.pos_y[start:j + 1]
                    X = [self.lattice_X[_x, _y] for _x, _y in zip(x, y)]
                    Y = [self.lattice_Y[_x, _y] for _x, _y in zip(x, y)]
                    self.lines[i].set_data(X, Y)
                    start = j + 1
                    i += 1
            else:
                x = s.pos_x[start:]
                y = s.pos_y[start:]
                X = [self.lattice_X[_x, _y] for _x, _y in zip(x, y)]
                Y = [self.lattice_Y[_x, _y] for _x, _y in zip(x, y)]
                self.lines[i].set_data(X, Y)
                i += 1

        if self.plot_surface:
            neighbors = []
            for bonding_pairs in self.bonding_pairs.values():
                # print(bonding_pairs)
                for pos in bonding_pairs.keys():
                    neighbors.append(pos)
            neighbors = list(np.array(neighbors).T)
            # print(neighbors)
            X, Y = self.lattice_X[neighbors], self.lattice_Y[neighbors]
            # print(X, Y)
            self.lines[-1].set_data(X, Y)


        # 最終的に，iの数だけ線を引けばよくなる
        # それ以上のオブジェクトはリセット
        if self.plot_surface:
            max_obj = len(self.lines) - 1
        else:
            max_obj = len(self.lines)
        for j in range(i, max_obj):
            self.lines[j].set_data([], [])

        return self.lines

    def update(self, num=0):
        """FuncAnimationから各フレームごとに呼び出される関数

        1時間ステップの間に行う計算はすべてここに含まれる。
        """

        # update each string
        for i, s in enumerate(self.strings): 
            if self.pre_function is not None:
                self.pre_func_res.append(self.pre_function(self, i, s))
            ret = self.update_each_string(i)
            if self.post_function is not None:
                self.post_func_res.append(self.post_function(self, i, s))

        if self.plot or self.save_video:
            ret = self.plot_string()
            return ret

    def update_each_string(self, key):
        X = self.get_neighbor_xy(key)
        if not X:
            raise StopIteration

        # print(X)
        s = self.strings[key]
        # update positions
        if len(X) == 4:
            i, r_rev, nx, ny = X
            s.x, s.y = nx, ny
            s.insert(0, r_rev)
            x, y = s.pos_x[0], s.pos_y[0]
        elif len(X) == 2:
            i, r = X
            s.insert(i + 1, r)
            x, y = s.pos_x[-1], s.pos_y[-1]
        else:
            i, r, r_rev = X
            s.vec[i] = r
            s.insert(i + 1, r_rev)
            x, y = s.pos_x[i + 1], s.pos_y[i + 1]

        self.occupied[x, y] = True

        # print("== start == (%d, %d)" % (x, y))

        # pp.pprint(self.bonding_pairs[key])

        for k, bonding_pairs in self.bonding_pairs.items():
            if bonding_pairs.has_key((x, y)):
                del self.bonding_pairs[k][(x, y)]
                # print("del self.bonding_pairs[%d][(%d, %d)]" % (k, x, y))

        # pp.pprint(self.bonding_pairs[key])

        if i == 0:
            # if s.loop:
            #     indexes = [[len(s.vec) - 2, len(s.vec)], [0, 2]]
            # else:
            # indexes = [[0, 2]]
            indexes = [[0, len(s.pos)]]
        elif i == len(s.pos) - 1:
            # if s.loop:
            #     indexes = [[len(s.vec) - 2, len(s.vec)], [0, 1]]
            # else:
                # indexes = [[len(s.vec) - 2, len(s.vec)]]
            indexes = [[max(0, len(s.pos)), len(s.pos)]]
        else:
            # indexes = [[i, len(s.pos)]]
            indexes = [[i, len(s.pos)]]

        self.cleanup_bonding_pairs(
            key=key,
            indexes=indexes
        )

        value = self.get_bonding_pairs(
            s=self.strings[key],
            indexes=indexes
        )


        # pp.pprint(value)
        # pp.pprint(self.bonding_pairs[key])

        for k, v in value.items():
            if self.bonding_pairs[key].has_key(k):
                self.bonding_pairs[key][k] += v
            else:
                self.bonding_pairs[key][k] = v
            # self.bonding_pairs[key] = value

        # pp.pprint(self.bonding_pairs[key])

        # print("== end ==")

        # pp.pprint(self.strings[key].pos)
        # pp.pprint(self.bonding_pairs[key].keys())


    def cleanup_bonding_pairs(self, key, indexes):
        rang = []
        for (start, stop) in indexes:
            rang += range(start, stop)
        for (x, y), l in self.bonding_pairs[key].items():
            tmp = []
            for i, (bonding_pair, w) in enumerate(l):
                if not bonding_pair[0] in rang:
                    tmp.append(l[i])
            if len(tmp) == 0:
                del self.bonding_pairs[key][(x, y)]
            else:
                self.bonding_pairs[key][(x, y)] = tmp

    def get_neighbor_xy(self, key):
        """Stringクラスのインスタンスsの隣接する非占有格子点の座標を取得する

        s (String): 対象とするStringクラスのインスタンス
        """
        if len(self.bonding_pairs[key]) == 0:
            return False

        # bonding_pairsの選ばれやすさを適切に重みを付けて評価
        weights = []
        bonding_pairs = []
        b = reduce(operator.add, self.bonding_pairs[key].values())
        # print(b)
        for (pair, w) in b:
            bonding_pairs.append(pair)
            weights.append(w)

        weights = np.array(weights)
        weights = weights / np.sum(weights)
        # print(weights)

        choiced_index = np.random.choice(range(len(weights)), p=weights)
        # print(bonding_pairs[choiced_index])
        return bonding_pairs[choiced_index]

    def calc_weight(self, s, i, r_i=None, r_rev=None):
        """ベクトルの内積を元に，Boltzmann分布に従って成長点選択の重みを決定
        """

        if (i == 1) and (not s.loop):
            w = self.dot(r_rev, s.vec[i]) - self.dot(s.vec[0], s.vec[1])
        elif (i == len(s.pos) - 1) and (not s.loop):
            w = self.dot(s.vec[i - 2], r_i) - self.dot(s.vec[i - 2], s.vec[i - 1])
        else:
            # w = self.dot(s.vec[i - 2], r_i) + self.dot(r_rev, s.vec[i % len(s.vec)])
            w = (self.dot(s.vec[i - 2], r_i) + self.dot(r_rev, s.vec[i % len(s.vec)])) \
                - (self.dot(s.vec[i - 2], s.vec[i - 1]) + self.dot(s.vec[i - 1], s.vec[i % len(s.vec)]))

        W = np.exp(self.beta * w)
        return W

    def get_bonding_pairs(self, s, indexes):
        bonding_pairs = {}
        neighbors_dict = {}
        rang = []
        for (start, stop) in indexes:
            rang += range(start, stop)

        if s.loop and (0 in rang):
            rang.append(0)

        for i in rang:
            x, y = s.pos[i]
            nnx, nny = self.lattice.neighbor_of(x, y)

            for r in [0, 1, 2, 3, 4, 5]:
                nx, ny = nnx[r], nny[r]
                if self.occupied[nx, ny] or nx == -1 or ny == -1:
                    continue

                r_rev = (r + 3) % 6

                if not neighbors_dict.has_key((nx, ny)):
                    if not s.loop:
                        if i == 0:
                            w = self.weight_const + self.dot(r_rev, s.vec[0])
                            W = np.exp(self.beta * w)
                            self.__update_dict(bonding_pairs,
                                               (nx, ny),
                                               [[0, r_rev, nx, ny], W])
                        elif i == len(s.pos) - 1:
                            w = self.dot(s.vec[i - 1], r) + self.weight_const
                            W = np.exp(self.beta * w)
                            self.__update_dict(bonding_pairs,
                                               (nx, ny),
                                               [[i, r], W])
                    neighbors_dict[(nx, ny)] = [(i, r),]
                else:
                    if neighbors_dict[(nx, ny)][-1][0] == i - 1:
                        r_i = neighbors_dict[(nx, ny)][-1][1]
                        w = self.calc_weight(s, i, r_i, r_rev)
                        self.__update_dict(bonding_pairs,
                                           (nx, ny),
                                           [[i - 1, r_i, r_rev], w])
                    neighbors_dict[(nx, ny)].append((i, r))
        return bonding_pairs


if __name__ == '__main__':
    # import timeit
    # print(timeit.timeit("Main(Lx=1000, Ly=1000, size=[3,] * 1, \
    #                           strings=[{'id': 1, 'x': 250, 'y': 500, 'vec': [0, 4]}], \
    #                           plot=False)",
    #                     setup="from __main__ import Main",
    #                     number=10
    #                     ))

    L = 100

    main= Main(Lx=L, Ly=L, size=[3,] * 1, frames=1000,
               beta=0.,
               plot=True, plot_surface=False,
               strings=[{'id': 1, 'x': L/4, 'y': L/2, 'vec': [0, 4]}])
