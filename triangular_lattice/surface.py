#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-09-12

import numpy as np


def get_surface_points(self, pos):
    surface_points = []
    for i, (x, y) in enumerate(pos):
        nnx, nny = self.lattice.neighborhoods[x, y]
        for r in [0, 1, 2, 3, 4, 5]:
            nx, ny = nnx[r], nny[r]
            if nx == -1 or ny == -1:
                continue
            elif self.occupied[nx, ny]:
                continue
            else:
                surface_points.append(i)
    pos_index = list(set(surface_points))
    return list(np.array(pos[pos_index]).T)


def set_labels(self, position):
    """Label the same number for points with connected (= cluster).
    """
    label = np.zeros([self.lattice.Lx, self.lattice.Ly], dtype=int)
    n = 1

    for i, j in np.array(position).T:
        nnx, nny = self.lattice.neighborhoods[i, j]
        # 6方向のラベルを参照
        tags = list(set([label[nnx[r], nny[r]]
                         for r in [0, 1, 2, 3, 4, 5]]) - set([0]))
        if len(tags) == 0:
            label[i, j] = n
            n += 1
        else:
            label[i, j] = min(tags)

    checked = []
    for i, j in reversed(np.array(np.where(label > 0)).T):
        if label[i, j] in checked:
            continue
        else:
            nnx, nny = self.lattice.neighborhoods[i, j]
            # 6方向のラベル + 自分自身を参照
            nn = (set([label[nnx[r], nny[r]] for r in [0, 1, 2, 3, 4, 5]])
                  | set([label[i, j]])
                 ) - set([0])

            min_tag = min(list(nn))
            for tag in nn - set([min_tag]):
                label[label == tag] = min_tag
                checked.append(tag)


    return label


def get_labeled_position(self, pos, test=False):
    position = get_surface_points(self, pos)
    label_lattice = set_labels(self, position)
    label_list = label_lattice[position]
    if test:
        # そのまま
        pos = position
    else:
        # 最大クラスターのみ抽出
        tag = np.argmax(np.bincount(label_list))
        pos = np.where(label_lattice == tag)
    return pos
