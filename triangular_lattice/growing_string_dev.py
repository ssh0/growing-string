#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-07-12


from __future__ import print_function
from triangular import LatticeTriangular as LT
from base import Main as base
from String import String
import numpy as np
import operator
import random
import pprint

pp = pprint.PrettyPrinter(indent=4)


class Main(base):
    """任意に設定したstringの近傍点に点を追加し成長させるモデル

    グラフ上で左端と右端に固定されたstringの近傍点を探索，ランダム(後には曲げ
    弾性による重み付けの効果を追加)に選択し，stringを成長させていくモデル
    """

    def __init__(self, Lx=40, Ly=40, boundary='periodic',
                 size=[5, 4, 10, 12], plot=True,
                 frames=1000,
                 dot_alpha=1.5,
                 dot_beta=1.,
                 weight_const=1.5,
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
        self.number_of_lines = sum(size) * Lx

        if strings is None:
            # Put the strings to the lattice
            self.strings = self.create_random_strings(len(size), size)
        else:
            self.strings = [String(self.lattice, **st) for st in strings]
        for string in self.strings:
            self.occupied[string.pos_x, string.pos_y] = True


        self.plot = plot
        self.interval = 1000
        self.frames = frames

        self.dot_result = self.create_dot_result(dot_alpha, dot_beta)
        self.weight_const = weight_const

        self.bonding_pairs = {i: {} for i in range(len(self.strings))}
        for key in self.bonding_pairs.keys():
            value = self.get_bonding_pairs(
                s=self.strings[key],
                indexes=[[0, len(self.strings[key].pos)]]
            )

            # TODO: 隣接点がないとき，全体のシミュレーションを終了する
            if len(value) == 0:
                return False

            self.bonding_pairs[key] = value

        pp.pprint(self.bonding_pairs)
        # return None

        self.pre_function = pre_function
        self.post_function = post_function
        self.pre_func_res = []
        self.post_func_res = []

        # Plot triangular-lattice points, string on it, and so on
        if self.plot:
            self.plot_all()
        else:
            t = 0
            while t < self.frames:
                try:
                    self.update()
                    t += 1
                except StopIteration:
                    break

    def create_dot_result(self, dot_alpha, dot_beta):
        """dot_alphaを高さとしてdot_resultの分布を決定する。
        """
        return [dot_alpha * (abs(3 - i) / 3.) ** dot_beta for i in range(6)]

    def dot(self, v, w):
        """0〜5で表された6つのベクトルの内積を計算する。

        v, w (int): ベクトル(0〜5の整数で表す)"""
        return self.dot_result[(w + 6 - v) % 6]

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

        if self.plot:
            ret = self.plot_string()
            return ret

    def update_each_string(self, key):
        X = self.get_neighbor_xy(key)
        if not X:
            raise StopIteration

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
            x, y = s.pos_x[i + 1], s.pos_y[i + 1]
        else:
            i, r, r_rev = X
            s.vec[i] = r
            s.insert(i + 1, r_rev)
            x, y = s.pos_x[i + 1], s.pos_y[i + 1]

        self.occupied[x, y] = True

        if self.bonding_pairs[key].has_key((x, y)):
            del self.bonding_pairs[key][(x, y)]
            print("del self.bonding_pairs[key][%d, %d]" % (x, y))


        if i == 0:
            if s.loop:
                indexes = [[len(s.vec - 1), len(s.vec)], [0, 2]]
            else:
                indexes = [[0, 2]]
        elif i == len(s.vec) - 1:
            if s.loop:
                indexes = [[len(s.vec - 2), len(s.vec)], [0, 1]]
            else:
                indexes = [[len(s.vec - 2), len(s.vec)]]
        else:
            indexes = [[i - 1, i + 2]]

        value = self.get_bonding_pairs(
            s=self.strings[key],
            indexes=indexes
        )

        for k, v in value.items():
            dict = self.bonding_pairs[key]
            if dict.has_key(k):
                dict[k] += v
            else:
                dict[k] = v
            # self.bonding_pairs[key] = value

        pp.pprint(self.bonding_pairs[key])

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

        choiced_index = np.random.choice(range(len(weights)), p=weights)
        return bonding_pairs[choiced_index]


    def __update_dict(self, dict, key, value):
        if dict.has_key(key):
            dict[key].append(value)
        else:
            dict[key] = [value]


    def get_bonding_pairs(self, s, indexes):
        bonding_pairs = {}
        neighbors_dict = {}
        rang = []
        for (start, stop) in indexes:
            rang += range(start, stop)

        for i in rang:
            x, y = s.pos[i]
            nnx, nny = self.lattice.neighborhoods[x, y]


            new_n = set(((nx, ny) for nx, ny in self.lattice.neighborhoods[x, y].T
                        if nx != -1 and ny != -1))
            updated_n = set(tuple(map(tuple, neighbors_dict))) | new_n
            self.neighbors = [pos for pos in updated_n if not self.occupied[pos]]


            for r in [0, 1, 2, 3, 4, 5]:
                nx, ny = nnx[r], nny[r]
                if self.occupied[nx, ny] or nx == -1 or ny == -1:
                    continue

                if neighbors_dict.get((nx, ny), False):
                    if neighbors_dict[(nx, ny)][-1][0] == i - 1:
                        r_i = neighbors_dict[(nx, ny)][-1][1]
                        r_rev = (r + 3) % 6

                        if i == 0:
                            w = self.dot(s.vec[i], r_i) + self.dot_result[0]
                        elif i == len(s.pos) - 1:
                            w = self.dot(r_rev, s.vec[i - 1]) + self.dot_result[0]
                        else:
                            w = self.dot(s.vec[i - 1], r_i) + \
                                self.dot(r_rev, s.vec[(i + 1) % len(s.vec)])

                        self.__update_dict(bonding_pairs, (nx, ny),
                                           ((i - 1, r_i, r_rev), w))
                    neighbors_dict[(nx, ny)].append((i, r))
                else:
                    r_rev = (r + 3) % 6
                    if i == 0:
                        w = self.weight_const + self.dot(r_rev, s.vec[0])

                        self.__update_dict(bonding_pairs, (nx, ny),
                                           ((0, r_rev, nx, ny), w))
                    elif i == len(pos) - 1:
                        w = self.weight_const + self.dot(s.vec[i - 1], r_rev)
                        self.__update_dict(bonding_pairs, (nx, ny),
                                           ((i, r_rev), w))

                    neighbors_dict[(nx, ny)] = [(i, r),]
        return bonding_pairs

    def calc_weight(self, s, bonding_pair):
        if len(bonding_pair) == 2:
            i, r = bonding_pair
            weight = self.weight_const + self.dot(s.vec[i - 1], r)
        elif len(bonding_pair) == 4:
            i, r_rev, nx, ny = bonding_pair
            weight = self.weight_const + self.dot(r_rev, s.vec[0])
        else:
            i, r, r_rev = bonding_pair
            if i == 0:
                weight = self.dot(s.vec[i], r) + self.dot_result[0]
            elif i == len(s.vec) - 1:
                weight = self.dot(r_rev, s.vec[i - 1]) + self.dot_result[0]
            else:
                weight = self.dot(s.vec[i - 1], r) + \
                    self.dot(r_rev, s.vec[i + 1])
        return weight



if __name__ == '__main__':
    # Ly = 40
    # main = Main(Lx=100, Ly=Ly, size=[Ly])

    # Ly = 5
    # main = Main(Lx=10, Ly=Ly, size=[Ly])

    # main = Main(Lx=6, Ly=6, size=[random.randint(4, 12)] * 1, plot=False)
    # main = Main(Lx=50, Ly=50, size=[random.randint(4, 12)] * 1, plot=False)
    # main = Main(Lx=30, Ly=30, size=[random.randint(4, 12) for i in range(3)])

    main = Main(Lx=60, Ly=60, size=[3,] * 1,
                strings=[{'id': 1, 'x': 30, 'y': 15, 'vec': [0, 4]}])

    # main = Main(Lx=60, Ly=60, size=[4,] * 1,
    #             strings=[{'id': 1, 'x': 30, 'y': 15, 'vec': [0, 4, 2]}])

