#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-04-22

import numpy as np
import random
from numpy.linalg import norm


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

    def get_distances(self, x, y):
        """Caluculate distance between two points and return list.

        --- Arguments ---
        x (list or ndarray): x座標の値のリスト
        y (list or ndarray): y座標の値のリスト
        """
        if self.is_open:
            distances = norm(np.c_[x[1:], y[1:]]
                             - np.c_[x[:-1], y[:-1]], axis=1)
            # len(distances) == self.N - 1
        else:
            distances = norm(np.c_[x, y]
                             - np.c_[np.roll(x, 1), np.roll(y, 1)], axis=1)
            # len(distances) == self.N

        return distances

    def grow(self, func_nl, func_k):
        """2点間の自然長を大きくする & バネ定数を変化させる

        バネ定数は単位(自然長)長さあたりが同じいなるように随時変化させる
        --- Arguments ---
        func_nl (function): N-1(開曲線)，N(閉曲線)次元のnp.arrayに対する関数
            返り値は同次元のnp.arrayで返し，これが成長後の自然長のリストである
        func_k (function): N-1(開曲線)，N(閉曲線)次元のnp.arrayに対する関数
            返り値は同次元のnp.arrayで返し，これが成長後のバネ定数のリスト
        """
        old_nl = self.natural_length
        new_nl = func_nl(self.natural_length)
        self.natural_length = new_nl
        self.K = func_k(self.K, old_nl, new_nl)

    def divide_if_extended(self, X):
        """2点間の自然長がlength_limitの設定値より大きいとき新しい点を追加する
        """
        j = 0
        for i in np.where(self.natural_length > self.length_limit)[0]:
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

        x_k, x_k1 = self.position_x[k], self.position_x[k + 1]
        y_k, y_k1 = self.position_y[k], self.position_y[k + 1]
        d_old = norm((x_k1 - x_k, y_k1 - y_k))

        # 点を追加
        new_positions = self.update_point_position(k, x_k, x_k1, y_k, y_k1)
        x_k, x_k1, x_k2, y_k, y_k1, y_k2 = new_positions

        # 距離を記録
        d_new_left = norm((x_k1 - x_k, y_k1 - y_k))
        d_new_right = norm((x_k2 - x_k1, y_k2 - y_k1))

        # 速度を更新
        self.update_point_velocity(k)

        # 自然長を更新
        nl_old, n_k1, n_k2 = self.update_natural_length(
            k, d_old, d_new_left, d_new_right)

        # バネ定数を更新
        self.update_spring_constant(k, d_old, d_new_left, d_new_right)

        self.N += 1

    def update_point_position(self, k, x_k, x_k1, y_k, y_k1):
        """点を追加

        Called from self.create_new_point
        Change: self.position_x, self.position_y
        """
        # 中点を返す場合
        pickp = lambda a, b: (b + a) / 2
        newpos_x = pickp(x_k, x_k1)
        newpos_y = pickp(y_k, y_k1)

        # 一様乱数で間の適当な値を選ぶ場合
        # scale = random.triangular(0.48, 0.52)
        # def pickp(a, b):
        #     return a + (b - a) * scale

        # newpos_x = pickp(x_k, x_k1)
        # newpos_y = pickp(y_k, y_k1)

        x_k2, y_k2 = x_k1, y_k1
        x_k1, y_k1 = newpos_x, newpos_y
        self.position_x = np.insert(self.position_x, k + 1, newpos_x)
        self.position_y = np.insert(self.position_y, k + 1, newpos_y)
        return x_k, x_k1, x_k2, y_k, y_k1, y_k2

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

    def update_natural_length(self, k, d_old, d_new_left, d_new_right):
        """自然長を更新

        Called from self.create_new_point
        Change: self.natural_length
        """
        nl_old = self.natural_length[k]

        # 長さに合わせて変化させる
        # n_k = nl_old * d_new_left / d_old
        # n_k1 = nl_old * d_new_right / d_old
        # self.natural_length[k] = n_k
        # self.natural_length = np.insert(self.natural_length, k + 1, n_k1)

        # 単純に2で割る
        n_k = self.natural_length[k] / 2
        n_k1 = n_k
        self.natural_length[k] = n_k
        self.natural_length = np.insert(self.natural_length, k + 1, n_k1)
        return nl_old, n_k, n_k1

    def update_spring_constant(self, k, nl_old, n_k, n_k1):
        """バネ定数を更新

        Called from self.create_new_point
        Change: self.K
        """
        # (元の定数kに対し，それぞれの長さ比で割ったものがバネ定数となる)
        # (↑分割に依ってエネルギーの総量は変わってはいけないという場合)
        # 今の場合，エネルギーは別に保存しなくても良い?
        new_k_left = self.K[k] * nl_old / n_k
        new_k_right = self.K[k] * nl_old/ n_k1
        self.K[k] = new_k_left
        self.K = np.insert(self.K, k + 1, new_k_right)

        # 単純に元のバネ定数を引き継ぐとした場合
        # new_k_left = self.K[k]
        # new_k_right = self.K[k]
        # self.K[k] = new_k_left
        # self.K = np.insert(self.K, k + 1, new_k_right)
