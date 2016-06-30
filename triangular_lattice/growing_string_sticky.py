#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-06-09

from triangular import LatticeTriangular as LT
from String import String
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.animation as animation
import numpy as np
from numpy import linalg as la
import random
import time

def print_debug(arg):
    """Print argument if needed.

    You can use this function in any parts and its behavior is toggled here.
    """
    # print arg
    pass


class Main:
    """両端を固定されたstringの近傍点に点を追加し成長させるモデル

    グラフ上で左端と右端に固定されたstringの近傍点を探索，ランダム(後には曲げ
    弾性による重み付けの効果を追加)に選択し，stringを成長させていくモデル
    """

    def __init__(self, Lx=40, Ly=40, lattice_scale=10.,
                 size=[5, 4, 10, 12], plot=True):
        """Init function of Main class.

        Lx (int (even)): 格子のx方向(グラフではy軸)の格子点数
        Ly (int (even)): 格子のy方向(グラフではx軸)の格子点数
        lattice_scale (int or float): グラフのx，y軸の実際のスケール(関係ない)
        """
        # Create triangular lattice with given parameters
        self.lattice = LT(np.zeros((Lx, Ly), dtype=np.int),
                          scale=lattice_scale, boundary='periodic')

        self.occupied = np.zeros((Lx, Ly), dtype=np.bool)
        self.number_of_lines = sum(size)

        # Put the strings to the lattice
        # self.strings = self.create_random_strings(N, size)
        # 今回は一本のstringを，Ly方向に伸ばした形で考える
        self.strings = [String(self.lattice, 1, int(Lx / 2), - int(Lx / 4) % Ly,
                               vec=[0] * (Ly - 1))]

        self.plot = plot

        # Plot triangular-lattice points, string on it, and so on
        if self.plot:
            self.plot_all()
        else:
            while True:
                try:
                    self.update()
                except StopIteration:
                    break

    def plot_all(self):
        """軸の設定，三角格子の描画，線分描画要素の用意などを行う

        ここからFuncAnimationを使ってアニメーション表示を行うようにする
        """
        frames = 1000
        self.fig, self.ax = plt.subplots(figsize=(8, 8))

        self.lattice_X = self.lattice.coordinates_x
        self.lattice_Y = self.lattice.coordinates_y

        X_min, X_max = min(self.lattice_X) - 0.1, max(self.lattice_X) + 0.1
        Y_min, Y_max = min(self.lattice_Y) - 0.1, max(self.lattice_Y) + 0.1
        self.ax.set_xlim([X_min, X_max])
        self.ax.set_ylim([Y_min, Y_max])
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])

        triang = tri.Triangulation(self.lattice_X, self.lattice_Y)
        self.ax.triplot(triang, color='#d5d5d5', marker='.', markersize=1)

        self.lines = [self.ax.plot([], [], marker='o', linestyle='-',
                                   color='black',
                                   markerfacecolor='black',
                                   markeredgecolor='black')[0]
                      for i in range(self.number_of_lines)]

        self.lattice_X = self.lattice_X.reshape(self.lattice.Lx,
                                                self.lattice.Ly)
        self.lattice_Y = self.lattice_Y.reshape(self.lattice.Lx,
                                                self.lattice.Ly)
        self.plot_string()

        # ani = animation.FuncAnimation(self.fig, self.update, frames=frames,
        #                               interval=1000, blit=True, repeat=False)
        plt.show()

    def plot_string(self):
        """self.strings内に格納されているStringを参照し，グラフ上に図示する
        """
        # print self.string.pos, self.string.vec

        i = 0  # to count how many line2D object
        for s in self.strings:
            start = 0
            for j, pos1, pos2 in zip(range(len(s.pos) - 1), s.pos[:-1], s.pos[1:]):
                dist_x = abs(self.lattice_X[pos1[0], pos1[1]] - self.lattice_X[pos2[0], pos2[1]] )
                dist_y = abs(self.lattice_Y[pos1[0], pos1[1]] - self.lattice_Y[pos2[0], pos2[1]] )
                # print j, pos1, pos2
                # print dist_x, dist_y
                if dist_x > 1.5 * self.lattice.dx or dist_y > 1.5 * self.lattice.dy:
                    x = s.pos_x[start:j+1]
                    y = s.pos_y[start:j+1]
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
        # 最終的に，iの数だけ線を引けばよくなる
        # それ以上のオブジェクトはリセット
        for j in range(i, len(self.lines)):
            self.lines[j].set_data([], [])

        return self.lines

    def update(self, num=0):
        """FuncAnimationから各フレームごとに呼び出される関数

        1時間ステップの間に行う計算はすべてここに含まれる。
        """
        # move head part of each strings (if possible)
        for s in self.strings:
            X = self.get_neighbor_xy(s)
            if not X:
                raise StopIteration

            # update starting position
            x, y, vec = X
            rmx, rmy = s.follow((x, y, (vec + 3) % 6))
            self.occupied[x, y] = True
            self.occupied[rmx, rmy] = False

        ret = self.plot_string()

        if self.plot:
            ret = self.plot_string()
            return ret

    def get_neighbor_xy(self, s):
        """Stringクラスのインスタンスsの隣接する非占有格子点の座標を取得する

        s (String): 対象とするStringクラスのインスタンス
        """
        neighbors_set = {}
        bonding_pairs = []
        # sのx, y座標に関して
        for i, (x, y) in enumerate(s.pos):
            # それぞれの近傍点を取得
            nnx, nny = self.lattice.neighborhoods[x, y]
            # 6方向全てに関して
            for r in range(6):
                # 反射境界条件のとき除外される点の場合，次の近接点に
                nx, ny = nnx[r], nny[r]
                if nx == -1 or ny == -1:
                    continue
                # 既に占有されているとき，次の近接点に
                if self.occupied[nx, ny]:
                    continue
                # それ以外(近傍点のうち占有されていない点であるとき)
                # 既にstringの近傍として登録されている場合
                if neighbors_set.has_key((nx, ny)):
                    # 一つ前に登録された点が現在の評価点の近傍点である場合
                    # -> bonding_pairsに[(前のxy), (s近傍のxy), (後のxy)]を追加
                    if neighbors_set[(nx, ny)][-1][0] == i - 1:
                        # 現在の評価点から近接点へのベクトルの逆向きのベクトル
                        r_rev = r + 3 % 6
                        # [i-1, [r_{i}, r_{rev}]]
                        bonding_pairs.append([i - 1,
                                              [neighbors_set[(nx, ny)][-1][1],
                                               r_rev]])
                    neighbors_set[(nx, ny)].append((i, r))
                # stringの近傍として登録されていない場合
                # -> 新たに登録
                else:
                    neighbors_set[(nx, ny)] = [(i, r), ]

        # この後やること ===
        # bonding_pairsの選ばれやすさを適切に重みを付けて評価
        ## さらに付け加えるなら ---
        ## 選ばれやすさはr_{i}とr_{rev}によって決まる角度によって決まる
        ## 重みつきでランダムに選択

        if len(vectors) == 0:
            print_debug("no neighbors")
            return False

        # 確率的に方向を決定
        vector = random.choice(vectors)
        # 点の格子座標を返す
        x, y = nnx[vector], nny[vector]
        return x, y, vector


if __name__ == '__main__':
    # main = Main()
    Ly = 40
    main = Main(Lx=100, Ly=Ly, size=[Ly])
