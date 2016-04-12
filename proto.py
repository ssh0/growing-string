#!/usr/bin/env python
# -*- coding:utf-8 -*-
""" Numerical simulation of growing strings

各点間をバネでつなぎ、その自然長が時間と共に増大することを考える。
この自然長、もしくは2点間の距離がある閾値を超えた時、新たに2点間に点を置く。
点に作用する力は、バネによる力と、曲げ弾性による力、粘性による摩擦力である。
オイラー法または4次のルンゲクッタ法でこれを解き、matplotlibでアニメーションに
して表示する。
"""
__author__ = "Shotaro Fujimoto"
__date__ = "2016/4/12"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# from scipy.spatial import euclidean as euc
import time
import logging as log
import random


class Euler(object):

    def __init__(self, function):
        """ Initialize function."""
        self.function = function

    def solve(self, y, t, h):
        """ Solve the system ODEs.

        --- arguments ---
        y: Array of initial values (ndarray)
        t: Time (float)
        h: Stepsize (float)
        """
        return y + h * self.function(t, y)


class RK4(object):

    def __init__(self, function):
        """ Initialize function."""
        self.function = function

    def solve(self, y, t, h):
        """ Solve the system ODEs.

        --- arguments ---
        y: Array of initial values (ndarray)
        t: Time (float)
        h: Stepsize (float)
        """
        k1 = h * self.function(t, y)
        k2 = h * self.function(t + h / 2, y + h * k1 / 2)
        k3 = h * self.function(t + h / 2, y + h * k2 / 2)
        k4 = h * self.function(t + h, y + h * k3 / 2)

        return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6


class Points():

    def __init__(self, N, position_x, position_y, natural_length, K,
                 length_limit):
        """Initialize class variants.

        --- Arguments ---
        N              (int)    : How many points should placed
        position_x     (ndarray): Array of the valuse of x axis for each points
        position_y     (ndarray): Array of the valuse of y axis for each points
        natural_length (ndarray): Array of natural length of each strings
        K              (ndarray): Array of spring constant
        length_limit   (float)  : Threshold for dividing to 2 strings
        """

        # 引数チェック
        if type(position_x) != np.ndarray or len(position_x) != N:
            raise UserWarning("expected %d dim ndarray for position_x" % N)
        if type(position_y) != np.ndarray or len(position_y) != N:
            raise UserWarning("expected %d dim ndarray for position_y" % N)

        # 開曲線か閉曲線か (開曲線: True)
        if len(natural_length) == N - 1:
            self.is_open = True
        elif len(natural_length) == N:
            self.is_open = False
        else:
            raise UserWarning(
                "expected N or N-1 dim ndarray for natural_length")

        if self.is_open:
            if len(K) != N - 1 or type(K) != np.ndarray:
                raise UserWarning("K must be ndarray with N-1 component")
        else:
            if len(K) != N or type(K) != np.ndarray:
                raise UserWarning("K must be ndarray with N component")

        if type(length_limit) != float:
            raise UserWarning("length_limit must be float")

        # 結節点の数
        self.N = N

        # 結節点の座標
        self.position_x = position_x
        self.position_y = position_y

        # 結節点の速度(0)
        self.vel_x = np.zeros(self.N)
        self.vel_y = np.zeros(self.N)

        # 各線分の自然長
        self.natural_length = natural_length
        # 開曲線の場合
        # len(self.natural_length) == N - 1
        # 閉曲線の場合
        # len(self.natural_length) == N
        # z = [z_{-1}, z_{0}, ... ]

        # バネ定数のリスト
        self.K = K
        # 開曲線の場合
        # len(self.K) == N - 1
        # 閉曲線の場合
        # len(self.K) == N
        # K = [k_{-1}, k_{0}, ... ]

        # 各バネに対する自然長の上限
        # これを超えるとその間に点を追加する
        self.length_limit = length_limit

    def get_distances(self, x_list, y_list):
        """Caluculate distance between two points and return list.

        --- Arguments ---
        x_list (list or ndarray): x座標の値のリスト
        y_list (list or ndarray): y座標の値のリスト
        """
        if self.is_open:
            distances = np.sqrt(np.power(x_list[1:] - x_list[:-1], 2) +
                                np.power(y_list[1:] - y_list[:-1], 2))
            # len(distances) == self.N - 1
        else:
            distances = np.sqrt(np.power(x_list - np.roll(x_list, 1), 2) +
                                np.power(y_list - np.roll(y_list, 1), 2))
            # len(distances) == self.N

        return distances

    def grow(self, func):
        """2点間の自然長を大きくする

        --- Arguments ---
        func (function): N-1(開曲線)，N(閉曲線)次元のnp.arrayに対する関数
            返り値は同次元のnp.arrayで返し，これが成長後の自然長のリストである
        """
        self.natural_length = func(self.natural_length)

    def divide_if_extended(self, X):
        """もし2点間距離がlength_limitの設定値より大きいとき，新しい点を追加する
        """

        j = 0
        for i in np.where(self.natural_length > self.length_limit)[0]:
            if self.is_open:
                k = i + j
            else:
                k = i + j - 1
            self.create_new_point(k, X)
            j += 1

        distances = self.get_distances(self.position_x, self.position_y)
        j = 0
        for i in np.where(distances > self.length_limit)[0]:
            if self.is_open:
                k = i + j
            else:
                k = i + j - 1
            self.create_new_point(k, X)
            j += 1

        return np.array([self.position_x, self.position_y,
                         self.vel_x, self.vel_y])

    def create_new_point(self, k, X):
        """新しい点を2点の間に追加し，各物理量を再設定する

        k番目とk+1番目の間に新しい点を追加
        """
        # 点を追加
        self.update_point_position(k)

        # 速度を更新
        self.update_point_velocity(k)

        d = self.get_distances(self.position_x, self.position_y)
        # 自然長を更新
        self.update_natural_length(k, d)

        # バネ定数を更新
        self.update_spring_constant(k)

        self.N += 1

    def update_point_position(self, k):
        """点を追加

        Called from self.create_new_point
        Change: self.position_x, self.position_y
        """
        # 中点を返す場合
        pickp = lambda a, b: (b + a) / 2

        # 一様乱数で間の適当な値を選ぶ場合
        # pickp = lambda a, b: (random.random() - 0.5) * (b - a) * 0.1 \
        #     + (b+a)/2

        newpos_x = pickp(self.position_x[k], self.position_x[k + 1])
        newpos_y = pickp(self.position_y[k], self.position_y[k + 1])
        self.position_x = np.insert(self.position_x, k + 1, newpos_x)
        self.position_y = np.insert(self.position_y, k + 1, newpos_y)

    def update_point_velocity(self, k):
        """速度を更新

        Called from self.create_new_point
        Change: self.vel_x, self.vel_y
        """
        # 新たに追加した点の速さは2点の平均とする
        # 運動量保存を考えて，元の点の速度も減少させる
        vel_xk = self.vel_x[k] - self.vel_x[k + 1] / 2
        vel_xkplus = self.vel_x[k + 1] - self.vel_x[k] / 2
        vel_xins = (self.vel_x[k] + self.vel_x[k + 1]) / 2
        self.vel_x[k] = vel_xk
        self.vel_x[k + 1] = vel_xkplus
        self.vel_x = np.insert(self.vel_x, k + 1, vel_xins)

        vel_yk = self.vel_y[k] - self.vel_y[k + 1] / 2
        vel_ykplus = self.vel_y[k + 1] - self.vel_y[k] / 2
        vel_yins = (self.vel_y[k] + self.vel_y[k + 1]) / 2
        self.vel_y[k] = vel_yk
        self.vel_y[k + 1] = vel_ykplus
        self.vel_y = np.insert(self.vel_y, k + 1, vel_yins)

    def update_natural_length(self, k, d):
        """自然長を更新

        Called from self.create_new_point
        Change: self.natural_length
        """
        # 長さに合わせて変化させる
        # if self.is_open:
        #     new_nl = d[k]
        #     self.natural_length[k] = new_nl
        #     new_nl = d[k+1]
        #     self.natural_length = np.insert(self.natural_length,
        #                                     k + 1, new_nl)
        # else:
        #     new_nl = d[k+1]
        #     self.natural_length[k] = new_nl
        #     new_nl = d[k+2]
        #     self.natural_length = np.insert(self.natural_length,
        #                                     k + 1, new_nl)

        new_nl = self.natural_length[k] / 2
        self.natural_length[k] = new_nl
        self.natural_length = np.insert(self.natural_length, k + 1, new_nl)

    def update_spring_constant(self, k):
        """バネ定数を更新

        Called from self.create_new_point
        Change: self.K
        """
        # (元の定数kに対し，それぞれの長さ比で割ったものがバネ定数となる)
        # (↑分割に依ってエネルギーの総量は変わってはいけないという場合)
        # 今の場合，エネルギーは別に保存しなくても良い?
        # if k == -1:
        #     d = self.get_distances(np.roll(self.position_x, 1)[0:3],
        #                            np.roll(self.position_y, 1)[0:3])
        # else:
        #     d = self.get_distances(self.position_x[k:k+3],
        #                            self.position_y[k:k+3])
        # new_k_left = self.K[k] * (d[0]/np.sum(d))
        # new_k_right = self.K[k] * (d[1]/np.sum(d))

        new_k_left = self.K[k]
        new_k_right = self.K[k]

        self.K[k] = new_k_left
        self.K = np.insert(self.K, k + 1, new_k_right)


class String_Simulation():

    def __init__(self, parameters):
        """Assign some initial values and parameters

        --- Arguments ---
        parameters (dict):
            key: x, y, nl, K, length_limit, h, t_max, e, debug_mode
            See details for each values in Points's documentation.
        """
        if type(params) != dict:
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
        self.grow_func = lambda arr: arr + 0.001
        # 乱数を含める
        # import random
        # def glow_randomly(arr):
        #     D = np.array([0.0005 + 0.001 * random.random()
        #                   for i in range(self.point.N)])
        #     return arr + D
        # self.grow_func = glow_randomly

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

    def force(self, t, X):  # TODO: unittest
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
            ee = 0.5 * self.e * (distances[1:] ** 5 + distances[:-1] ** 5) / 2
            ee = np.insert(ee, 0, 0)
            ee = np.append(ee, 0)
        else:
            # 閉曲線の場合
            ee = 0.5 * self.e * (distances ** 5. +
                                 np.roll(distances, -1) ** 5.)
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

        # 粘性項D
        return np.array([X[2],
                         X[3],
                         np.dot(Z, X[0]) + np.dot(B, X[0]) - self.D * X[2],
                         np.dot(Z, X[1]) + np.dot(B, X[1]) - self.D * X[3]
                         ])

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

        self.t, count, frame = 0., 0, 0
        # solver = RK4(self.force)  # Runge-Kutta method
        solver = Euler(self.force)  # Euler method
        while self.t < self.t_max:
            if not self.pause:
                log.info(self.t)
                X = solver.solve(X, self.t, self.h)
                # update values
                self.point.position_x, self.point.position_y = X[0], X[1]
                self.point.vel_x, self.point.vel_y = X[2], X[3]

                # 一定の周期で各バネの自然長を増加させる
                if self.t > 0.01 * (count + 1):  # TODO: 要検討
                    self.point.grow(self.grow_func)
                    count += 1

                # 各点間の距離が基準値を超えていたら，間に新たな点を追加する
                # 自然長が超えている場合も同じ様にするべき?
                X = self.point.divide_if_extended(X)

                # 一定の間隔で描画を行う
                if self.t > self.h * 9 * frame:  # TODO: 要検討
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
                self.t += self.h
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
            print "t: " + str(self.t)
            print "x: " + str(self.point.position_x)
            print "y: " + str(self.point.position_y)
            print "d: " + str(self.point.get_distances(self.point.position_x,
                                                       self.point.position_y))
            print "nl: " + str(self.point.natural_length)
            print "K: " + str(self.point.K)

        # ctrl+p キーでPause
        if event.key == "ctrl+p":
            self.pause_simulation()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Simulation of growing strings')
    parser.add_argument('--debug', dest='debug_mode', action='store_true',
                        help='debug mode: print some verbose information')
    args = parser.parse_args()

    # initial values
    params = {
        "open 4": {
            'x': np.array([-2., 0., 2., 4.]),
            'y': np.array([0., -0.01, 0.01, 0.]),
            'nl': np.array([2., 2., 2.]),
            'K': np.array([300., 300., 300.]),
            'length_limit': 4.,
            'h': 0.005,
            't_max': 130.,
            'e': 0.08,
            'D': 10.,
            'debug_mode': args.debug_mode
        },

        "close 4": {
            'x': np.array([0., 1., 1., 0.]),
            'y': np.array([0., 0., 1., 1.]),
            'nl': np.array([1., 1., 1., 1.]),
            'K': np.array([300., 300., 300., 300.]),
            'length_limit': 4.,
            'h': 0.005,
            't_max': 130.,
            'e': 0.08,
            'D': 10.,
            'debug_mode': args.debug_mode
        },

        "close 3": {
            'x': np.array([0., 1.73, 1.73]),
            'y': np.array([0., 1., -1.]),
            'nl': np.array([2., 2., 2.]),
            'K': np.array([300., 300., 300.]),
            'length_limit': 4.,
            'h': 0.005,
            't_max': 130.,
            'e': 0.08,
            'D': 10.,
            'debug_mode': args.debug_mode
        },
    }
    sim = String_Simulation(params['close 4'])
    sim.run()
