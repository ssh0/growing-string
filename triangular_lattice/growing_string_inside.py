#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-10-21


from triangular import LatticeTriangular as LT
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.animation as animation
import numpy as np


class InsideString(object):
    def __init__(self, Lx=40, Ly=40,
                 boundary={'h': 'periodic', 'v': 'periodic'},
                 initial_state=[(20, 20)],
                 plot=True,
                 plot_surface=True,
                 save_image=False,
                 save_video=False,
                 filename_image="",
                 filename_video="",
                 frames=1000,
                 beta = 2.,
                 interval=1,
                 pre_function=None,
                 post_function=None):
        """Init function of the class"""
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

        self.kagome_Lx = 2 * self.lattice.Lx
        self.kagome_Ly = self.lattice.Ly
        x_even = self.lattice_X + 0.5 * self.lattice.dx
        y_even = self.lattice_Y + self.lattice.dy / 3.
        x_odd = np.roll(self.lattice_X, -1, axis=0)
        y_odd = np.roll(self.lattice_Y, -1, axis=0) + (2 * self.lattice.dy) / 3.
        self.kagome_X = np.hstack((x_even, x_odd)).reshape(self.kagome_Lx,
                                                           self.kagome_Ly)
        self.kagome_Y = np.hstack((y_even, y_odd)).reshape(self.kagome_Lx,
                                                           self.kagome_Ly)

        self.occupied = np.zeros((self.kagome_Lx, self.kagome_Ly),
                                 dtype=np.bool)
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
        self.pre_function = pre_function
        self.post_function = post_function
        self.pre_func_res = []
        self.post_func_res = []

        self.beta = beta

        self.weight_rule = (
            (0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 1),
            (0, 0, 0, 0, 0, 0, 0, 1, 1),
            (0, 0, 0, 0, 0, 0, 1, 1, 1),
            (1, 0, 0, 0, 0, 0, 0, 0, 0),
            (1, 0, 0, 0, 0, 0, 0, 0, 1),
            (1, 0, 0, 0, 0, 0, 0, 1, 1),
            (1, 0, 0, 0, 0, 0, 1, 1, 1),
            (1, 1, 0, 0, 0, 0, 0, 0, 0),
            (1, 1, 0, 0, 0, 0, 0, 0, 1),
            (1, 1, 0, 0, 0, 0, 0, 1, 1),
            (1, 1, 0, 0, 0, 0, 1, 1, 1),
            (1, 1, 1, 0, 0, 0, 0, 0, 0),
            (1, 1, 1, 0, 0, 0, 0, 0, 1),
            (1, 1, 1, 0, 0, 0, 0, 1, 1),
            (1, 1, 1, 0, 0, 0, 1, 1, 1)
        )
        self.weight_list = np.array([
            -2., -1.5, -0.5, 0.,
            -1.5, -1., 0., 0.5,
            -0.5, 0., 1., 1.5,
            0., 0.5, 1.5, 2.
        ])

        self.weight_list = np.exp(- self.beta * self.weight_list)
        self.weight_table = {k: v for k, v in
                             zip(self.weight_rule, self.weight_list)}
        # for k, w in self.weight_table.items():
        #     print str(k) + ' : ' + str(w)
        self.growing_points = {}  # {(x, y): weight}

        # initial state
        for pos in initial_state:
            self.occupied[pos] = True
            self.append_new_growing_point(pos)
        self.initial_state = initial_state

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
            # print("Image file is successfully saved at '%s'." % filename_image)
            plt.close()

    def append_new_growing_point(self, pos):
        """新たに追加された点の格子座標posを元に，その周辺の非占有点の座標と
        占有確率を取得してself.growing_pointsに追加する関数

        --- Arguments ---
        pos: tuple (pos_x, pos_y): 新たに追加された点の格子座標

        --- Modify ---
        self.growing_points: dict {(x, y): weight}:
        """
        pos_x, pos_y = pos
        if pos_x % 2 == 0:
            even_or_odd = 'even'
        elif pos_x % 2 == 1:
            even_or_odd = 'odd'

        # 新たに追加された点の第一近傍を取得
        nn = getattr(self, 'get_nn1_' + even_or_odd)(pos_x, pos_y)
        for k, (x, y) in nn.items():
            # 既に占有されていたら飛ばす
            if self.occupied[(x, y)]:
                continue
            # その点の第一近傍を取得
            if even_or_odd == 'even':
                even_or_odd_ = 'odd'
            elif even_or_odd == 'odd':
                even_or_odd_ = 'even'
            nn1 = getattr(self, 'get_nn1_' + even_or_odd_)(x, y)
            nn1 = {i: pos for i, pos in nn1.items() if self.occupied[pos]}
            # 第一近傍がひとつだけ占有されている場合以外は飛ばす
            if len(nn1) != 1:
                continue
            # この時点で
            # nn1 = {'1': (x1, y1)}
            # のようになる

            # 点の周りの第二近傍を取得する
            nn2 = getattr(self, 'get_nn2_' + even_or_odd_)(x, y)
            # nn2 = {'4': (x4, y4), ...}
            # 真偽表を作成
            truth_table = self.create_truth_table(nn1.keys()[0], nn2)
            # 真偽表から(x, y)に関する重みを取得，growing_pointsに追加
            if truth_table in self.weight_rule:
                self.growing_points[(x, y)] = self.weight_table[truth_table]
        # print self.growing_points

    def cleanup_growing_point(self):
        for pos in self.growing_points.keys():
            pos_x, pos_y = pos
            if pos_x % 2 == 0:
                even_or_odd = 'even'
            elif pos_x % 2 == 1:
                even_or_odd = 'odd'

            nn1 = getattr(self, 'get_nn1_' + even_or_odd)(pos_x, pos_y)
            nn1 = {i: pos for i, pos in nn1.items() if self.occupied[pos]}
            if len(nn1) != 1:
                del self.growing_points[(pos_x, pos_y)]
                continue

            nn2 = getattr(self, 'get_nn2_' + even_or_odd)(pos_x, pos_y)
            i = nn1.keys()[0]
            if i == '1':
                index = ['7', '8', '9']
            elif i == '2':
                index = ['10', '11', '12']
            elif i == '3':
                index = ['4', '5', '6']

            if not tuple(self.occupied[nn2[i]] for i in index) == (0, 0, 0):
                del self.growing_points[(pos_x, pos_y)]

    def create_truth_table(self, i, nn2):
        """nn1のkeyにしたがって，ルールへの対応に変換

        --- Arguments ---
        i (str): 追加候補点の第一近傍点の相対位置を表すインデックス
        nn2 (dict) {i: (x, y)}: i:追加候補点の第2近傍点のインデックス
        """
        if i == '1':
            index = ['4', '5', '6', '7', '8', '9', '10', '11', '12']
        elif i == '2':
            index = ['7', '8', '9', '10', '11', '12', '4', '5', '6']
        elif i == '3':
            index = ['10', '11', '12', '4', '5', '6', '7', '8', '9']

        return tuple(self.occupied[nn2[i]] for i in index) 

    def get_nn1_even(self, x, y):
        """格子座標(x, y)の第一近傍の点の座標を返す(xが偶数の時)"""
        return {
            '1': (x - 1 % self.kagome_Lx, y),
            '2': (x + 1, y),
            '3': (x + 1, y - 1 % self.kagome_Ly),
        }

    def get_nn1_odd(self, x, y):
        """格子座標(x, y)の第一近傍の点の座標を返す(xが奇数の時)"""
        return {
            '1': (x - 1, y),
            '2': (x - 1, y + 1 % self.kagome_Ly),
            '3': (x + 1 % self.kagome_Lx, y),
        }

    def get_nn2_even(self, x, y):
        """格子座標(x, y)の第二近傍の点の座標を返す(xが偶数の時)"""
        return {
            '4':  (x - 2 % self.kagome_Lx, y + 1 % self.kagome_Ly),
            '5':  (x - 1 % self.kagome_Lx, y + 1 % self.kagome_Ly),
            '6':  (x, y + 1 % self.kagome_Ly),
            '7':  (x + 2 % self.kagome_Lx, y),
            '8':  (x + 3 % self.kagome_Lx, y - 1 % self.kagome_Ly),
            '9':  (x + 2 % self.kagome_Lx, y - 1 % self.kagome_Ly),
            '10': (x, y - 1 % self.kagome_Ly),
            '11': (x - 1 % self.kagome_Lx, y - 1 % self.kagome_Ly),
            '12': (x - 2 % self.kagome_Lx, y)
        }

    def get_nn2_odd(self, x, y):
        """格子座標(x, y)の第二近傍の点の座標を返す(xが奇数の時)"""
        return {
            '4':  (x - 2 % self.kagome_Lx, y),
            '5':  (x - 3 % self.kagome_Lx, y + 1 % self.kagome_Ly),
            '6':  (x - 2 % self.kagome_Lx, y + 1 % self.kagome_Ly),
            '7':  (x, y + 1 % self.kagome_Ly),
            '8':  (x + 1 % self.kagome_Lx, y + 1 % self.kagome_Ly),
            '9':  (x + 2 % self.kagome_Lx, y),
            '10': (x + 2 % self.kagome_Lx, y - 1 % self.kagome_Ly),
            '11': (x + 1 % self.kagome_Lx, y - 1 % self.kagome_Ly),
            '12': (x, y - 1 % self.kagome_Ly)
        }

    def plot_all(self):
        """軸の設定，三角格子の描画，線分描画要素の用意などを行う

        ここからFuncAnimationを使ってアニメーション表示を行うようにする
        """
        self.fig, self.ax = plt.subplots(figsize=(8, 8))

        lattice_X = self.lattice.coordinates_x
        lattice_Y = self.lattice.coordinates_y
        X_min, X_max = min(lattice_X) - 0.1, max(lattice_X) + 0.1
        Y_min, Y_max = min(lattice_Y) - 0.1, max(lattice_Y) + 0.1
        self.ax.set_xlim([X_min, X_max])
        self.ax.set_ylim([Y_min, Y_max])
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.set_aspect('equal')

        triang = tri.Triangulation(lattice_X, lattice_Y)
        self.ax.triplot(triang, color='#d5d5d5', marker='.', markersize=1)

        self.points = [self.ax.plot([], [], 'g.')[0]]
        if self.plot_surface:
            self.points.append(self.ax.plot([], [], '.', color='#ff0000')[0])

        self.plot_points()

    def start_animation(self, filename=""):
        if self.__dict__.has_key('frames'):
            frames = self.frames
        else:
            frames = 1000

        def init_func(*arg):
            return self.points

        ani = animation.FuncAnimation(self.fig, self.update, frames=frames,
                                      init_func=init_func,
                                      interval=self.interval,
                                      blit=True, repeat=False)
        if filename != "":
            ani.save(filename, codec="libx264", bitrate=-1, fps=30)
            # try:
                # ani.save(filename, codec="libx264", bitrate=-1, fps=30)
            # except:
            #     print("Can't saved.")
            # else:
            #     print("Animation is successfully saved at '%s'." % filename)
        else:
            plt.show()

    def plot_points(self):
        """self.occupiedを，グラフ上に図示する
        self.plot_surfaceが指定されている時には，成長点もプロットする
        """
        pos = np.where(self.occupied)
        # pos = np.array([pos_x, pos_y]).T.tolist()
        X, Y = self.kagome_X[pos], self.kagome_Y[pos]

        self.points[0].set_data(X, Y)

        if self.plot_surface:
            pos_x, pos_y = np.array(self.growing_points.keys()).T
            # pos = list(np.array(pos_x, pos_y).T)
            X, Y = self.kagome_X[pos_x, pos_y], self.kagome_Y[pos_x, pos_y]
            self.points[1].set_data(X, Y)

        return self.points

    def update(self, num=0):
        """FuncAnimationから各フレームごとに呼び出される関数

        1時間ステップの間に行う計算はすべてここに含まれる。
        """

        if len(self.growing_points) == 0:
            print "no neighbors"
            raise StopIteration

        if self.pre_function is not None:
            self.pre_func_res.append(self.pre_function(self))

        positions = self.growing_points.keys()
        weights = np.array([self.growing_points[key] for key in positions])
        weights = weights / np.sum(weights)
        # print(weights)

        index = np.random.choice(range(len(positions)), p=weights)
        x, y = positions[index]
        self.occupied[x, y] = True
        self.cleanup_growing_point()
        self.append_new_growing_point((x, y))
        del self.growing_points[(x, y)]

        if self.post_function is not None:
            self.post_func_res.append(self.post_function(self))

        if self.plot or self.save_video:
            return self.plot_points()


if __name__ == '__main__':
    L = 60
    frames = 1000
    beta = 0.

    params = {
        'Lx': L,
        'Ly': L,
        'frames': frames,
        'beta': beta,
        'boundary': {'h': 'periodic', 'v': 'periodic'},
        # 'boundary': {'h': 'reflective', 'v': 'reflective'},
        'plot': True,
        'plot_surface': True,
        'interval': 1,
    }

    main = InsideString(initial_state=[(L / 2, L / 2 - 1)], **params)

