#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2017-03-03
"""論文[1]に従い六角格子内の1点をその六角格子セルを代表する点とみなし，
隣接する6つのセルを代表する点を繋ぐことでランダムな三角格子を生成する
[1] https://www.jstage.jst.go.jp/article/journalcpij/44.3/0/44.3_799/_pdf
"""

import numpy as np


def pick_param():
    """Pick up random point from hex region"""
    # -1 <= p <= 1
    p = 2. * np.random.rand() - 1.
    # -1 <= q <= 1
    q = 2. * np.random.rand() - 1.
    if p+q >= -1 and p+q <= 1:
        return p, q
    else:
        return pick_param()

def randomize(lattice):
    X, Y = lattice.coordinates_x, lattice.coordinates_y

    ab = np.array([pick_param() for n in range(X.shape[0])])
    xy = np.dot(ab,
                np.array([
                    [1., 0.],
                    [0.5, np.sqrt(3)/2]
                ])
                )
    X, Y = X + 0.5 * lattice.dx * xy.T[0], Y + 0.5 * lattice.dx * xy.T[1]
    lattice.coordinates_x, lattice.coordinates_y = X, Y

    return X, Y

if __name__ == '__main__':
    from triangular import LatticeTriangular as LT
    import matplotlib.pyplot as plt
    import matplotlib.tri as tri


    fig, ax = plt.subplots()

    Lx, Ly = 50, 30
    # Lx, Ly = 10, 10
    lattice = LT(
        np.zeros((Lx, Ly), dtype=np.int),
        scale=float(max(Lx, Ly)),
        boundary={'h': 'periodic',
                  'v': 'reflective'}
    )

    lattice_X = lattice.coordinates_x
    lattice_Y = lattice.coordinates_y
    X_min, X_max = min(lattice_X) - 0.7, max(lattice_X) + 0.7
    Y_min, Y_max = min(lattice_Y) - 0.5, max(lattice_Y) + 0.5
    ax.set_xlim([X_min, X_max])
    ax.set_ylim([Y_min, Y_max])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')

    _triang = tri.Triangulation(lattice_X, lattice_Y)
    ax.scatter(lattice_X, lattice_Y, s=5)

    randomize(lattice)

    triang = tri.Triangulation(lattice.coordinates_x, lattice.coordinates_y, triangles=_triang.triangles)
    # triang = tri.Triangulation(lattice.coordinates_x, lattice.coordinates_y)
    # ax.triplot(triang, color='#d5d5d5', lw=0.5)
    ax.triplot(triang, color='k', lw=0.5)


    # trilattice.lattice[neighbors] = 2
    # colorseq = np.zeros((Lx, Ly))
    # colorseq[trilattice.lattice == 2] = 0.9
    # colorseq[trilattice.lattice == 0] = 0.
    # colorseq[trilattice.lattice == 1] = 0.5

    # X, Y = trilattice.to_realspace(scale=20, x0=-10, y0=-10)
    # import matplotlib.pyplot as plt
    # plt.scatter(X, Y, s=100., c=colorseq)
    # plt.show()

    plt.show()
