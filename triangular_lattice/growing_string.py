#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-07-12


from triangular import LatticeTriangular as LT
from base import Main as base
from String import String
import numpy as np
import random

def print_debug(arg):
    """Print argument if needed.

    You can use this function in any parts and its behavior is toggled here.
    """
    # print arg
    pass


class Main(base):
    """任意に設定したstringの近傍点に点を追加し成長させるモデル

    グラフ上で左端と右端に固定されたstringの近傍点を探索，ランダム(後には曲げ
    弾性による重み付けの効果を追加)に選択し，stringを成長させていくモデル
    """

    def __init__(self, Lx=40, Ly=40, lattice_scale=10.,
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
        lattice_scale (int or float): グラフのx，y軸の実際のスケール(関係ない)
        """
        # Create triangular lattice with given parameters
        self.lattice = LT(np.zeros((Lx, Ly), dtype=np.int),
                          scale=float(max(Lx, Ly)), boundary='periodic')

        self.lattice_X = self.lattice.coordinates_x.reshape(self.lattice.Lx,
                                                self.lattice.Ly)
        self.lattice_Y = self.lattice.coordinates_y.reshape(self.lattice.Lx,
                                                self.lattice.Ly)

        self.occupied = np.zeros((Lx, Ly), dtype=np.bool)
        self.number_of_lines = sum(size) * Lx

        if strings is None:
            # Put the strings to the lattice
            self.strings = self.create_random_strings(len(size), size)
            # self.strings = [String(lattice=self.lattice, id=1,
            #                        x=int(Lx / 2), y=-int(Lx / 4) % Ly,
            #                        vec=[0] * ((Ly - 1) / 2) + [1] 
            #                        + [3] * ((Ly - 1) / 2) + [4])]
        else:
            self.strings = [String(self.lattice, **st) for st in strings]
        self.occupied[self.strings[0].pos_x, self.strings[0].pos_y] = True

        self.plot = plot
        self.interval = 1
        self.frames = frames

        self.dot_result = self.create_dot_result(dot_alpha, dot_beta)
        self.weight_const = weight_const

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
            ret = self.update_each_string(s)
            if self.post_function is not None:
                self.post_func_res.append(self.post_function(self, i, s))

        if self.plot:
            ret = self.plot_string()
            return ret

    def update_each_string(self, s):
        X = self.get_neighbor_xy(s)
        if not X:
            raise StopIteration

        # update positions
        if len(X) == 4:
            i, r_rev, nx, ny = X
            s.x, s.y = nx, ny
            s.insert(0, r_rev)
            self.occupied[s.pos_x[0], s.pos_y[0]] = True
        elif len(X) == 2:
            i, r = X
            s.insert(i + 1, r)
            self.occupied[s.pos_x[i + 1], s.pos_y[i + 1]] = True
        else:
            i, r, r_rev = X
            s.vec[i] = r
            s.insert(i + 1, r_rev)
            self.occupied[s.pos_x[i + 1], s.pos_y[i + 1]] = True

    def get_neighbor_xy(self, s):
        """Stringクラスのインスタンスsの隣接する非占有格子点の座標を取得する

        s (String): 対象とするStringクラスのインスタンス
        """
        self.neighbors_set = {}
        bonding_pairs = self.get_bonding_pairs(s)
        if len(bonding_pairs) == 0:
            return False
        choosed_pair = self.choose_one_bonding_pair(s, bonding_pairs)
        return choosed_pair

    def get_bonding_pairs(self, s):
        bonding_pairs = []
        # sのx, y座標に関して
        for i, (x, y) in enumerate(s.pos):
            # それぞれの近傍点を取得
            nnx, nny = self.lattice.neighborhoods[x, y]
            # 6方向全てに関して
            for r in [0, 1, 2, 3, 4, 5]:
                nx, ny = nnx[r], nny[r]
                # 反射境界条件のとき除外される点の場合，次の近接点に
                if nx == -1 or ny == -1:
                    continue
                # 既に占有されているとき，次の近接点に
                elif self.occupied[nx, ny]:
                    continue
                # それ以外(近傍点のうち占有されていない点であるとき)
                # 既にstringの近傍として登録されている場合
                elif self.neighbors_set.has_key((nx, ny)):
                    # 一つ前に登録された点が現在の評価点の近傍点である場合
                    if self.neighbors_set[(nx, ny)][-1][0] == i - 1:
                        bonding_pair = [i - 1,
                                        self.neighbors_set[(nx, ny)][-1][1],
                                        (r + 3) % 6]
                        # [i-1, r_{i}, r_{rev}]
                        # r_rev: 現在の点から近接点へのベクトル
                        bonding_pairs.append(bonding_pair)
                    self.neighbors_set[(nx, ny)].append((i, r))
                # stringの近傍として登録されていない場合 -> 新たに登録
                else:
                    if i == 0:
                        # r_rev = (r + 3) % 6
                        bonding_pairs.append([0, (r + 3) % 6, nx, ny])
                    if i == len(s.pos) - 1:
                        bonding_pairs.append([i, r])
                    self.neighbors_set[(nx, ny)] = [(i, r),]
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

    def choose_one_bonding_pair(self, s, bonding_pairs):
        # bonding_pairsの選ばれやすさを適切に重みを付けて評価
        weights = np.array([self.calc_weight(s, p) for p in bonding_pairs])
        weights = weights / np.sum(weights)

        choiced_index = np.random.choice(range(len(weights)), p=weights)
        return bonding_pairs[choiced_index]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # Ly = 40
    # main = Main(Lx=100, Ly=Ly, size=[Ly])

    # Ly = 5
    # main = Main(Lx=10, Ly=Ly, size=[Ly])

    # main = Main(Lx=6, Ly=6, size=[random.randint(4, 12)] * 1, plot=False)
    # main = Main(Lx=50, Ly=50, size=[random.randint(4, 12)] * 1, plot=False)
    main = Main(Lx=50, Ly=50, size=[random.randint(4, 12)] * 1)
