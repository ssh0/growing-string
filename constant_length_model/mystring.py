#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-04-22

import numpy as np
import random
from numpy.linalg import norm
from scipy.spatial import distance

class MyString:

    def __init__(self, N, position_x, position_y, natural_length, K, is_open,
                 update_point_position_model='midpoint'):
        """Initialize class variants.

        --- Arguments ---
        N              (int)    : How many points should placed
        position_x     (ndarray): Array of the valuse of x axis for each points
        position_y     (ndarray): Array of the valuse of y axis for each points
        natural_length (float)  : Natural length of all strings
        K              (float)  : Spring constant of all strings

        update_point_position_model (str): Give the model how to choose
            the position of the new point.
            ('midpoint' or 'random_uniform' or 'random_triangular')
        """

        #
        if type(position_x) != np.ndarray or len(position_x) != N:
            raise UserWarning("expected %d dim ndarray for position_x" % N)
        if type(position_y) != np.ndarray or len(position_y) != N:
            raise UserWarning("expected %d dim ndarray for position_y" % N)

        self.is_open = is_open
        # 開曲線か閉曲線か (開曲線: True)
        # if len(natural_length) == N - 1:
        #     self.is_open = True
        # elif len(natural_length) == N:
        #     self.is_open = False
        # else:
        #     raise UserWarning(
        #         "expected N or N-1 dim ndarray for natural_length")

        if type(K) != float:
            raise UserWarning("K must be float")

        # 結節点の数
        self.N = N

        # 結節点の座標
        self.position_x = position_x
        self.position_y = position_y

        # 結節点の速度(0)
        self.vel_x = np.zeros(self.N)
        self.vel_y = np.zeros(self.N)

        # 自然長
        self.natural_length = natural_length

        # バネ定数
        self.K = K

        self.update_point_position_model = update_point_position_model

    def distances(self, x, y):
        """Caluculate distance between two points and return list.

        --- Arguments ---
        x (list or ndarray): x座標の値のリスト
        y (list or ndarray): y座標の値のリスト
        """
        if self.is_open:
            res = norm(np.c_[x[1:], y[1:]] - np.c_[x[:-1], y[:-1]], axis=1)
            # len == self.N - 1
        else:
            res = norm(np.c_[x, y] - np.c_[np.roll(x, 1), np.roll(y, 1)],
                       axis=1)
            # len == self.N

        return res

    def distance_matrix(self):
        """Matrix of the distances between two points."""
        xx = np.c_[self.position_x, self.position_y]
        matrix = distance.cdist(xx, xx)

        return matrix

    def add(self):
        """点kと点k+1の間に新たな点を配置し，線素を1つ増やす"""
        k = random.randint(-1, self.N - 2)
        self.create_new_point_at(k)
        return [self.position_x, self.position_y, self.vel_x, self.vel_y]

    def create_new_point_at(self, k):
        """新しい点を2点間に作成し，各物理量を再設定する"""
        x_k, x_k1 = self.position_x[k], self.position_x[k + 1]
        y_k, y_k1 = self.position_y[k], self.position_y[k + 1]
        # d_old = norm((x_k1 - x_k, y_k1 - y_k))

        # 点を追加
        new_positions = self.update_point_position(k, x_k, x_k1, y_k, y_k1)
        x_k, x_k1, x_k2, y_k, y_k1, y_k2 = new_positions

        # 距離を記録
        # d_new_left = norm((x_k1 - x_k, y_k1 - y_k))
        # d_new_right = norm((x_k2 - x_k1, y_k2 - y_k1))

        # 速度を更新
        self.update_point_velocity(k)

        self.N += 1

    def update_point_position(self, k, x_k, x_k1, y_k, y_k1):
        """点を追加

        Called from self.create_new_point
        Change: self.position_x, self.position_y
        """
        if self.update_point_position_model == 'midpoint':
            # 中点を返す場合
            pickp = lambda a, b: (b + a) / 2
            newpos_x = pickp(x_k, x_k1)
            newpos_y = pickp(y_k, y_k1)
        elif self.update_point_position_model == 'random_uniform':
            # 一様乱数で間の適当な値を選ぶ場合
            scale = random.uniform(0.48, 0.52)
            pickp = lambda a, b: a + (b - a) * scale
            newpos_x = pickp(x_k, x_k1)
            newpos_y = pickp(y_k, y_k1)
        elif self.update_point_position_model == 'random_triangular':
            scale = random.triangular(0.48, 0.52)
            pickp = lambda a, b: a + (b - a) * scale
            newpos_x = pickp(x_k, x_k1)
            newpos_y = pickp(y_k, y_k1)

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

