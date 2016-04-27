#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-04-22

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import logging as log
import random
import time
from euler import Euler
from runge_kutta import RK4
from points import Points


class String_Simulation():

    def __init__(self, parameters):
        """Assign some initial values and parameters

        --- Arguments ---
        parameters (dict):
            key: x, y, nl, K, length_limit, h, t_max, e, debug_mode
            See details for each values in Points's documentation.
        """
        if type(parameters) != dict:
            raise TypeError("params should be dictionary")
        self.__dict__ = parameters

        # debugging
        if self.debug_mode:
            log.basicConfig(format="%(levelname)s: %(message)s",
                            level=log.DEBUG)
        else:
            log.basicConfig(format="%(levelname)s: %(message)s")

        self.point = Points(N=len(self.x),
                            position_x=self.x,
                            position_y=self.y,
                            natural_length=self.nl,
                            K=self.K,
                            length_limit=self.length_limit)

        # 成長率の決定
        # 累積的に成長する場合
        # self.grow_func = lambda arr: arr * 1.01
        # 一定の成長率で成長する場合
        # self.grow_func = lambda arr: arr + 0.00025
        self.grow_func = lambda arr: arr + 0.001
        # 乱数を含める
        # import random
        # def glow_randomly(arr):
        #     D = np.array([0.0005 + 0.001 * random.random()
        #                   for i in range(self.point.N)])
        #     return arr + D
        # self.grow_func = glow_randomly

        def grow_func_k(arr, old_nl, new_nl):
            return arr * old_nl / new_nl

        self.grow_func_k = grow_func_k

        self.fig, self.ax = plt.subplots()
        self.l, = self.ax.plot(np.array([]), np.array([]), 'bo-')
        self.ax.set_xlim([-30., 30.])
        self.ax.set_ylim([-30., 30.])

        # toggle pause/resume by clicking the matplotlib canvas region
        self.pause = False

    def run(self):
        # Launch onClick by button_press_event
        self.fig.canvas.mpl_connect('button_press_event', self.onClick)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        # In order to increase fps, decrease 'interval' arguments
        # (default value of animation.TimedAnimation is 200)
        # ani = animation.FuncAnimation(self.fig, self.animate, self.update,
        #                               interval=300, blit=True, repeat=False)
        ani = animation.FuncAnimation(self.fig, self.animate, self.update,
                                      interval=0, blit=True, repeat=False)

        # -- Save to mp4 file
        # -- Set up formatting for the movie files
        # TODO: 何故かシミュレーションの途中で終わってしまう。
        #       要調査
        # Writer = animation.writers['ffmpeg']
        # writer = Writer(fps=25, metadata=dict(artist='Me'), bitrate=192000)
        # now = time.asctime()
        # ani.save("output_"+now+".mp4", writer=writer)

        # Show the window
        plt.show()

    def animate(self, data):
        """FuncAnimationから呼ぶ。ジェネレータupdateから返された配列を描画する
        """
        self.l.set_data(data[0], data[1])
        return self.l,

    def __force_create_matrix(self, t, X):
        """self.forceとself.force_with_more_viscosityで用いられる行列を計算

        バネ弾性，曲げ弾性による力を計算するための変換行列を生成する。
        """
        distances = self.point.get_distances(X[0], X[1])

        # 圧縮由来の力を表す行列Zを作成
        if self.point.is_open:
            # 開曲線のとき，先頭はゼロにする
            zz = np.insert(self.point.natural_length / distances - 1., 0, 0)
            K = np.insert(self.point.K, 0, 0)
        else:
            # 閉曲線の場合
            zz = self.point.natural_length / distances - 1.
            K = self.point.K
        # どちらの場合でも len(zz) = N

        zz = np.diag(zz)
        K = np.diag(K)
        z = np.dot(K, zz)
        zl = -np.roll(z, -1, axis=1)
        zu = -np.roll(z, -1, axis=0)
        zul = -np.roll(zu, -1, axis=1)
        Z = z + zl + zu + zul

        # 曲げ由来の力を表す行列Bを作成
        if self.point.is_open:
            # 開曲線のとき，先頭,末尾はゼロにする
            ee = self.e * (distances[1:] ** (-3) + distances[:-1] ** (-3)) / 2
            ee = np.insert(ee, 0, 0)
            ee = np.append(ee, 0)
        else:
            # 閉曲線の場合
            ee = self.e * (distances ** (-3) +
                           np.roll(distances, -1) ** (-3)) / 2
        # どちらの場合でも len(ee) = N
        ee = np.diag(ee)
        el = np.roll(ee, -1, axis=1)
        er = np.roll(ee, 1, axis=1)
        eu = np.roll(ee, -1, axis=0)
        ed = np.roll(ee, 1, axis=0)
        eru = np.roll(er, -1, axis=0)
        erd = np.roll(er, 1, axis=0)
        elu = np.roll(el, -1, axis=0)
        eld = np.roll(el, 1, axis=0)
        B = -(eld + erd + elu + eru) / 4 + (el + ed + er + eu) / 2 - ee

        return Z, B

    def force(self, t, X):
        """各点にかかる力を，バネ弾性と曲げ弾性，粘性の効果を入れて計算

        1タイムステップでの変化
        @return X'
        X = [x, y, x', y']
        X' = [x', y', f_x/m, f_y/m]
        """
        # 必要な変換行列を計算
        Z, B = self.__force_create_matrix(t, X)
        # 粘性項D, 質量m
        return np.array([X[2],
                         X[3],
                         (np.dot(Z, X[0]) + np.dot(B, X[0]) - self.D * X[2])
                         / self.m,
                         (np.dot(Z, X[1]) + np.dot(B, X[1]) - self.D * X[3])
                         / self.m
                         ])

    def force_with_more_viscosity(self, t, X):
        """関数forceの変化形。粘性が優位な場合

        1 タイムステップでの変化
        @return X'
        X = [x, y, x', y']
        粘性が非常に大きい時，運動方程式
        mx'' = f - D v
        のx''は殆ど無視できるとして，式を簡単にすると
        D v = f
        の式を解くことになる。(x''の項は考慮しなくて良くなる)
        X' = [f_x/D, f_y/D, dummy. dummy]
        """
        # 必要な変換行列を計算
        Z, B = self.__force_create_matrix(t, X)

        # 粘性項D
        return np.array([(np.dot(Z, X[0]) + np.dot(B, X[0])) / self.D,
                         (np.dot(Z, X[1]) + np.dot(B, X[1])) / self.D,
                         X[2], X[3]
                         ])

    def update_position_self_avoiding(self):
        """自己回避効果を取り入れ，他の線分要素との重なりを解消する

        cross_detect関数を用いて交差判定を行い，座標の交換を行う
        NOTE: 簡単に無限ループに入って抜け出せなくなるので，
        別のアプローチを採るべき
        TODO: 運動方程式内でそれぞれの線分要素に短距離斥力ポテンシャルを与える
        """
        crossing = True
        # while crossing:
        count = 0
        lis = range(self.point.N - 1)
        # random.shuffle(lis)
        for i in lis:
            llis = range(i + 2, self.point.N - 2)
            # random.shuffle(llis)
            for k in llis:
                x_i = self.point.position_x[i]
                y_i = self.point.position_y[i]
                x_i1 = self.point.position_x[i+1]
                y_i1 = self.point.position_y[i+1]
                x_k = self.point.position_x[k]
                y_k = self.point.position_y[k]
                x_k1 = self.point.position_x[k+1]
                y_k1 = self.point.position_y[k+1]
                if self.cross_detect(x_i, x_i1, x_k, x_k1,
                                        y_i, y_i1, y_k, y_k1):
                    # Update positions
                    # distance_i_k1 = norm(np.array([x_i - x_k1, y_i - y_k1]))
                    # distance_i1_k = norm(np.array([x_i1 - x_k, y_i1 - y_k]))
                    distance_i_k1 = abs(x_i - x_k1) + abs(y_i - y_k1)
                    distance_i1_k = abs(x_i1 - x_k) + abs(y_i1 - y_k)
                    if distance_i_k1 > distance_i1_k:
                        self.point.position_x[i+1] = 0.75 * x_k + 0.25 * x_i1
                        self.point.position_y[i+1] = 0.75 * y_k + 0.25 * y_i1
                        self.point.position_x[k] = 0.25 * x_k + 0.75 * x_i1
                        self.point.position_y[k] = 0.25 * y_k + 0.75 * y_i1
                        # 速度を反転
                        self.point.vel_x[i+1] = - self.point.vel_x[i+1]
                        self.point.vel_y[i+1] = - self.point.vel_y[i+1]
                        self.point.vel_x[k] = - self.point.vel_x[k]
                        self.point.vel_y[k] = - self.point.vel_y[k]

                    else:
                        self.point.position_x[i] = 0.75 * x_k1 + 0.25 * x_i
                        self.point.position_y[i] = 0.75 * y_k1 + 0.25 * y_i
                        self.point.position_x[k+1] = 0.25 * x_k1 + 0.75 * x_i
                        self.point.position_y[k+1] = 0.25 * y_k1 + 0.75 * y_i
                        # 速度を反転
                        self.point.vel_x[i] = - self.point.vel_x[i]
                        self.point.vel_y[i] = - self.point.vel_y[i]
                        self.point.vel_x[k+1] = - self.point.vel_x[k+1]
                        self.point.vel_y[k+1] = - self.point.vel_y[k+1]
                    # self.point.position_x[i+1] = 0.55 * x_k + 0.45 * x_i1
                    # self.point.position_y[i+1] = 0.55 * y_k + 0.45 * y_i1
                    # self.point.position_x[k] = 0.45 * x_k + 0.55 * x_i1
                    # self.point.position_y[k] = 0.45 * y_k + 0.55 * y_i1
                    # self.point.position_x[i+1] = x_k
                    # self.point.position_y[i+1] = y_k
                    # self.point.position_x[k] = x_i1
                    # self.point.position_y[k] = y_i1

                    count += 1
            if count == 0:
                crossing = False

    def cross_detect(self, x1, x2, x3, x4, y1, y2, y3, y4):
        """2つの線分の交差判定を行う

        @return True/False
        線分1: (x1, y1), (x2,y2)
        線分2: (x3, y3), (x4,y4)
        # 線分が接する場合にはFalseを返すこととする
        """
        # まず,絶対に交差しない場合を除く
        # xについて
        if x1 < x2:
            if (x3 < x1 and x4 < x1) or (x3 > x2 and x4 > x2):
                return False
        else:
            if (x3 < x2 and x4 < x2) or (x3 > x1 and x4 > x1):
                return False

        # yについて
        if y1 < y2:
            if (y3 < y1 and y4 < y1) or (y3 > y2 and y4 > y2):
                return False
        else:
            if (y3 < y2 and y4 < y2) or (y3 > y1 and y4 > y1):
                return False

        if ((x1 - x2)*(y3 - y1) + (y1 - y2)*(x1 - x3)) * \
        ((x1 - x2)*(y4 - y1) + (y1 - y2)*(x1 - x4)) >= 0:
            return False

        if ((x3 - x4)*(y1 - y3) + (y3 - y4)*(x3 - x1)) * \
        ((x3 - x4)*(y2 - y3) + (y3 - y4)*(x3 - x2)) >= 0:
            return False

        # Else
        return True

    def update(self):
        """時間発展(タイムオーダーは成長よりも短くすること)

        各点にかかる力は，それぞれに付いているバネから受ける力の合力。
        Runge-Kutta法を用いて運動方程式を解く。
        この内部でglow関数を呼ぶ

        --- Arguments ---
        point (class): 参照するPointクラスを指定する
        h     (float): シミュレーションの時間発展の刻み
        t_max (float): シミュレーションを終了する時間
        """

        # 初期条件
        X = np.array([self.point.position_x, self.point.position_y,
                      self.point.vel_x, self.point.vel_y
                      ])
        # X = [[x0, x1, ... , xN-1],
        #      [y1, y2, ... , yN-1],
        #      [x'0, x'1, ... , x'N-1],
        #      [y'1, y'2, ..., y'N-1]]

        # solver = RK4(self.force)  # Runge-Kutta method
        # solver = RK4(self.force_with_more_viscosity)  # Runge-Kutta method
        # solver = Euler(self.force)  # Euler method
        solver = Euler(self.force_with_more_viscosity)  # Euler method

        self.t, t_count, frame = 0., 0, 0
        while self.t < self.t_max:
            if not self.pause:
                X = solver.solve(X, self.t, self.h)
                # update values
                self.point.position_x, self.point.position_y = X[0], X[1]
                self.point.vel_x, self.point.vel_y = X[2], X[3]

                # 各バネの自然長を増加させる & バネ定数を変化させる
                self.point.grow(self.grow_func, self.grow_func_k)

                # 各点間の距離が基準値を超えていたら，間に新たな点を追加する
                X = self.point.divide_if_extended(X)

                # self avoiding
                if self.self_avoiding:
                    self.update_position_self_avoiding()

                # 一定の間隔で描画を行う
                if self.t > self.h * 12 * frame:  # TODO: 要検討
                    log.info(self.t)
                    log.info("N: " + str(self.point.N))
                    log.info("x: " + str(self.point.position_x))
                    log.info("y: " + str(self.point.position_y))
                    log.info("d: " + str(self.point.get_distances(
                        self.point.position_x, self.point.position_y)))
                    log.info("nl: " + str(self.point.natural_length))
                    log.info("K: " + str(self.point.K))
                    if self.point.is_open:
                        yield [self.point.position_x, self.point.position_y]
                    else:
                        yield [np.append(self.point.position_x,
                                         self.point.position_x[0]),
                               np.append(self.point.position_y,
                                         self.point.position_y[0])]
                    frame += 1
                t_count += 1
                self.t = self.h * t_count
            else:
                time.sleep(0.1)
                if self.point.is_open:
                    yield [self.point.position_x, self.point.position_y]
                else:
                    yield [np.append(self.point.position_x,
                                     self.point.position_x[0]),
                           np.append(self.point.position_y,
                                     self.point.position_y[0])]
        print "Done!"

    def pause_simulation(self):
        """シミュレーションを一時停止"""
        if self.pause:
            print "Resume the simulation ..."
        else:
            print "[Paused] Please click the figure to resume the simulation"
        self.pause ^= True

    def onClick(self, event):
        """matplotlibの描画部分をマウスでクリックすると一時停止"""
        self.pause_simulation()

    def on_key(self, event):
        """キーを押すことでシミュレーション中に動作"""

        # i キーで情報表示
        if event.key == "i":
            print "--- [information] ---"
            print "x: " + str(self.point.position_x)
            print "y: " + str(self.point.position_y)
            print "d: " + str(self.point.get_distances(self.point.position_x,
                                                       self.point.position_y))
            print "nl: " + str(self.point.natural_length)
            print "K: " + str(self.point.K)
            print "t: " + str(self.t)
            print "N: " + str(self.point.N)

        # ctrl+p キーでPause
        if event.key == "ctrl+p":
            self.pause_simulation()
