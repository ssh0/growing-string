#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-10-21


from triangular import LatticeTriangular as LT
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.animation as animation
import networkx as nx
# import pygraphviz
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
                 record_networks=False,
                 frames=1000,
                 beta = 2.,
                 interval=1,
                 pre_function=None,
                 post_function=None):
        """Init function of the class"""

        self.plot = plot
        self.plot_surface = plot_surface
        self.save_image = save_image
        self.save_video = save_video
        self.record_networks = record_networks
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
        self.beta = beta
        self.pre_function = pre_function
        self.post_function = post_function
        self.pre_func_res = []
        self.post_func_res = []

        self.init(Lx, Ly, boundary, initial_state)
        self.start()

    def init(self, Lx, Ly, boundary, initial_state):
        self.lattice = LT(
            np.zeros((Lx, Ly), dtype=np.int),
            scale=float(max(Lx, Ly)),
            boundary=boundary
        )
        if self.record_networks:
            self.G = nx.Graph()

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
        y_even = self.lattice_Y + self.lattice.dy / 2.
        x_odd = np.roll(self.lattice_X, -1, axis=0)
        y_odd = np.roll(self.lattice_Y, -1, axis=0) + (2 * self.lattice.dy) / 3.
        self.kagome_X = np.hstack((x_even, x_odd)).reshape(self.kagome_Lx,
                                                           self.kagome_Ly)
        self.kagome_Y = np.hstack((y_even, y_odd)).reshape(self.kagome_Lx,
                                                           self.kagome_Ly)
        self.occupied = np.zeros((self.kagome_Lx, self.kagome_Ly),
                                 dtype=np.bool)

        self._create_weight_table()

        self.growing_points = {}  # {(x, y): weight}

        # initial state
        for pos in initial_state:
            self.occupied[pos] = True
            self.append_new_growing_point(pos)
        if self.record_networks:
            self.G.add_nodes_from(map(str, initial_state))
        self.initial_state = initial_state

    def start(self):
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

    def _create_weight_table(self):
        """Create the rule table."""

        _weight_rule = (
            (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # 2048
            (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),  # 2049
            (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1),  # 2051
            (1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1),  # 2055
            (1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0),  # 2304
            (1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1),  # 2305
            (1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1),  # 2307
            (1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1),  # 2311
            (1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0),  # 2432
            (1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1),  # 2433
            (1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1),  # 2435
            (1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1),  # 2439
            (1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0),  # 2496
            (1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1),  # 2497
            (1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1),  # 2499
            (1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1),  # 2503
            (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # 1024
            (0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0),  # 1088
            (0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0),  # 1216
            (0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0),  # 1472
            (0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0),  # 1056
            (0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0),  # 1120
            (0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0),  # 1248
            (0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0),  # 1504
            (0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0),  # 1072
            (0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0),  # 1136
            (0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0),  # 1264
            (0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0),  # 1520
            (0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0),  # 1080
            (0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0),  # 1144
            (0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0),  # 1272
            (0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0),  # 1528
            (0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # 512
            (0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0),  # 520
            (0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0),  # 536
            (0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0),  # 568
            (0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0),  # 516
            (0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0),  # 524
            (0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0),  # 540
            (0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0),  # 572
            (0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0),  # 518
            (0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0),  # 526
            (0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0),  # 542
            (0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0),  # 574
            (0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1),  # 519
            (0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1),  # 527
            (0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1),  # 543
            (0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1),  # 575
        )

        _weight_list = np.array([
            # -2., -1.5, -0.5, 0.,
            # -1.5, -1., 0., 0.5,
            # -0.5, 0., 1., 1.5,
            # 0., 0.5, 1.5, 2.,
            # -2., -1.5, -0.5, 0.,
            # -1.5, -1., 0., 0.5,
            # -0.5, 0., 1., 1.5,
            # 0., 0.5, 1.5, 2.,
            # -2., -1.5, -0.5, 0.,
            # -1.5, -1., 0., 0.5,
            # -0.5, 0., 1., 1.5,
            # 0., 0.5, 1.5, 2.,

            -1.5, -1., 0., 0.5,
            -1., -0.5, 0.5, 1.,
            0., 0.5, 1.5, 2.,
            0.5, 1., 2., 2.5,
            -1.5, -1., 0., 0.5,
            -1., -0.5, 0.5, 1.,
            0., 0.5, 1.5, 2.,
            0.5, 1., 2., 2.5,
            -1.5, -1., 0., 0.5,
            -1., -0.5, 0.5, 1.,
            0., 0.5, 1.5, 2.,
            0.5, 1., 2., 2.5,
        ])


        self.weight_rule = [int(''.join(map(str, t)), 2) for t in _weight_rule]
        self.weight_list = np.exp(- self.beta * _weight_list)
        self.weight_table = {k: v for k, v in
                             zip(self.weight_rule, self.weight_list)}

    def _create_truth_table(self, i, nn2):
        truth_table = [0, 0, 0]
        truth_table[int(i) - 1] = 1
        truth_table += [1 if self.occupied[_nn2] else 0 for _nn2 in nn2]
        truth_table = int(''.join(map(str, truth_table)), 2)
        return truth_table

    def append_new_growing_point(self, pos):
        """新たに追加された点の格子座標posを元に，その周辺の非占有点の座標と
        占有確率を取得してself.growing_pointsに追加する関数

        --- Arguments ---
        pos: tuple (pos_x, pos_y): 新たに追加された点の格子座標

        --- Modify ---
        self.growing_points: dict {(x, y): weight}:
        """
        pos_x, pos_y = pos
        even_or_odd = 'even' if pos_x % 2 == 0 else 'odd'

        # 新たに追加された点の第一近傍を取得
        nn = getattr(self, 'get_nn1_' + even_or_odd)(pos_x, pos_y)
        for k, (x, y) in nn.items():
            # 既に占有されていたら飛ばす
            if self.occupied[(x, y)]:
                continue
            # その点の第一近傍を取得
            even_or_odd_ = 'odd' if even_or_odd == 'even' else 'even'
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
            # 真偽表から(x, y)に関する重みを取得，growing_pointsに追加
            truth_table = self._create_truth_table(nn1.keys()[0], nn2)
            if truth_table in self.weight_rule:
                self.growing_points[(x, y)] = self.weight_table[truth_table]

    def cleanup_growing_point(self):
        for pos in self.growing_points.keys():
            pos_x, pos_y = pos
            even_or_odd = 'even' if pos_x % 2 == 0 else 'odd'
            nn1 = getattr(self, 'get_nn1_' + even_or_odd)(pos_x, pos_y)
            nn1 = {i: pos for i, pos in nn1.items() if self.occupied[pos]}
            if len(nn1) != 1:
                del self.growing_points[(pos_x, pos_y)]
                continue

            nn2 = getattr(self, 'get_nn2_' + even_or_odd)(pos_x, pos_y)
            truth_table = self._create_truth_table(nn1.keys()[0], nn2)
            if not truth_table in self.weight_rule:
                del self.growing_points[(pos_x, pos_y)]

    def get_nn1_even(self, x, y):
        """格子座標(x, y)の第一近傍の点の座標を返す(xが偶数の時)"""
        return {
            '1': ((x - 1) % self.kagome_Lx, y),
            '2': ((x + 1) % self.kagome_Lx, y),
            '3': ((x + 1) % self.kagome_Lx, (y - 1) % self.kagome_Ly),
        }

    def get_nn1_odd(self, x, y):
        """格子座標(x, y)の第一近傍の点の座標を返す(xが奇数の時)"""
        return {
            '1': ((x - 1) % self.kagome_Lx, y),
            '2': ((x - 1) % self.kagome_Lx, (y + 1) % self.kagome_Ly),
            '3': ((x + 1) % self.kagome_Lx, y),
        }

    def get_nn2_even(self, x, y):
        """格子座標(x, y)の第二近傍の点の座標を返す(xが偶数の時)"""
        return [
            ((x - 2) % self.kagome_Lx, (y + 1) % self.kagome_Ly),
            ((x - 1) % self.kagome_Lx, (y + 1) % self.kagome_Ly),
            (x, (y + 1) % self.kagome_Ly),
            ((x + 2) % self.kagome_Lx, y),
            ((x + 3) % self.kagome_Lx, (y - 1) % self.kagome_Ly),
            ((x + 2) % self.kagome_Lx, (y - 1) % self.kagome_Ly),
            (x, (y - 1) % self.kagome_Ly),
            ((x - 1) % self.kagome_Lx, (y - 1) % self.kagome_Ly),
            ((x - 2) % self.kagome_Lx, y)
        ]

    def get_nn2_odd(self, x, y):
        """格子座標(x, y)の第二近傍の点の座標を返す(xが奇数の時)"""
        return [
            ((x - 2) % self.kagome_Lx, y),
            ((x - 3) % self.kagome_Lx, (y + 1) % self.kagome_Ly),
            ((x - 2) % self.kagome_Lx, (y + 1) % self.kagome_Ly),
            (x, (y + 1) % self.kagome_Ly),
            ((x + 1) % self.kagome_Lx, (y + 1) % self.kagome_Ly),
            ((x + 2) % self.kagome_Lx, y),
            ((x + 2) % self.kagome_Lx, (y - 1) % self.kagome_Ly),
            ((x + 1) % self.kagome_Lx, (y - 1) % self.kagome_Ly),
            (x, (y - 1) % self.kagome_Ly)
        ]

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

        # self.points = [self.ax.plot([], [], 'g^')[0],
        #                self.ax.plot([], [], 'gv')[0]]
        self.points = [self.ax.plot([], [], 'k.')[0],
                       self.ax.plot([], [], 'k.')[0]]
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
        pos_even = (pos[0][pos[0] % 2 == 0], pos[1][pos[0] % 2 == 0])
        pos_odd = (pos[0][pos[0] % 2 == 1], pos[1][pos[0] % 2 == 1])
        X, Y = self.kagome_X[pos_even], self.kagome_Y[pos_even]
        self.points[0].set_data(X, Y)
        X, Y = self.kagome_X[pos_odd], self.kagome_Y[pos_odd]
        self.points[1].set_data(X, Y)

        if self.plot_surface:
            pos_x, pos_y = np.array(self.growing_points.keys()).T
            # pos = list(np.array(pos_x, pos_y).T)
            X, Y = self.kagome_X[pos_x, pos_y], self.kagome_Y[pos_x, pos_y]
            self.points[2].set_data(X, Y)

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
        if self.record_networks:
            self.G.add_node('({}, {})'.format(x, y))
            even_or_odd = 'even' if x % 2 == 0 else 'odd'
            nn1 = getattr(self, 'get_nn1_' + even_or_odd)(x, y)
            for pos in nn1.values():
                if self.occupied[pos]:
                    self.G.add_edge(str(pos), '({}, {})'.format(x, y))

        self.cleanup_growing_point()
        self.append_new_growing_point((x, y))
        del self.growing_points[(x, y)]

        if self.post_function is not None:
            self.post_func_res.append(self.post_function(self))

        if self.plot or self.save_video:
            return self.plot_points()


if __name__ == '__main__':
    L = 100
    frames = 1000
    beta = 4.

    params = {
        'Lx': L,
        'Ly': L,
        'frames': frames,
        'beta': beta,
        'boundary': {'h': 'periodic', 'v': 'periodic'},
        # 'boundary': {'h': 'reflective', 'v': 'reflective'},
        'plot': True,
        'plot_surface': False,
        'record_networks': False,
        'interval': 1,
    }

    main = InsideString(initial_state=[(L / 2, L / 2 - 1)], **params)

