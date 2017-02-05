#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-10-16


from diecutting_hexagonal import DieCuttingHexagonal
import numpy as np
import itertools
from tqdm import tqdm
import time
import argparse

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import save_data


# Remark: diecutting.pyのときは1つのLに対して複数の長方形領域を取得していたので，
#         関数を2回作用させてやる必要があったが，このシミュレーションの場合，
#         1つのLに対して，サブクラスターのリストは1つだけ取得されるので，
#         関数を二つ指定する必要はない。
result_set = {
    # 'num_of_sub_clusters': {
    #     'func': len,
    # },
    'size_dist_of_sub_clusters': {
        'func': lambda arr: np.bincount(map(len, arr)),
    },
    # 'size_dist_ave_of_sub_clusters': {
    #     'func': lambda arr: np.bincount(map(len, arr)),
    # },
    # 'max_size_of_sub_cluster': {
    #     'func': lambda arr: max(map(len, arr)),
    # }
}

def eval_simulation_for_each_string(params):
    main = DieCuttingHexagonal(params)
    main.start(result_set, visualize=False)
    # return main.cutting_sizes, main.res['num_of_sub_clusters']
    return main.cutting_sizes, main.res

def eval_simulation_for_one_beta(beta, num_of_strings=30):
    current_time = time.strftime("%y%m%d_%H%M%S")
    frames = 1000
    params = {
        'L': (frames + 2) * 2,
        'frames': frames,
        'beta': beta,
        'plot': False
    }

    d = {k: {} for k in result_set.keys()}
    for s in tqdm(range(num_of_strings)):
        # _Ls, _N_sub = eval_simulation_for_each_string(params)
        _Ls, _res = eval_simulation_for_each_string(params)
        for k in result_set.keys():
            for i, l in enumerate(_Ls):
                if d[k].has_key(l):
                    d[k][l].append(_res[k][i])
                else:
                    d[k][l] = [_res[k][i],]

    if d.has_key('num_of_sub_clusters'):
        # # 以下のやり方だと，Lが存在しないサンプルに対して無視した結果となる
        # mean = [(k, np.average(v)) for k, v in d.items()]
        # カットサイズLが存在しない場合には0で置き換えたような平均のとり方
        mean = [(k, np.sum(v) / float(num_of_strings))
                for k, v in d['num_of_sub_clusters'].items()]
        Ls, N_sub = np.array(sorted(mean)).T
    else:
        N_sub = []

    if d.has_key('size_dist_of_sub_clusters'):
        size_dist = {}
        for k, v in d['size_dist_of_sub_clusters'].items():
            # size_dist[k] = map(sum, itertools.izip_longest(*v, fillvalue=0))
            size_dist[k] = map(sum, itertools.izip_longest(*v, fillvalue=1))
        Ls = sorted(size_dist.keys())
        size_dist = [size_dist[k] for k in Ls]
        S = np.zeros((len(size_dist), max(map(len, size_dist))))
        for i, s in enumerate(size_dist):
            for j, num in enumerate(s):
                S[i][j] = num
        size_dist = S
    else:
        size_dist = []

    import numpy.ma as ma
    def masked_average(arr):
        return ma.array(arr, mask=np.array(arr) == -1).mean()

    if d.has_key('size_dist_ave_of_sub_clusters'):
        size_dist_ave = {}
        for k, v in d['size_dist_ave_of_sub_clusters'].items():
            num_when_L = itertools.izip_longest(*v, fillvalue=-1)
            size_dist_ave[k] = map(masked_average, num_when_L)
        Ls = sorted(size_dist_ave.keys())
        size_dist_ave = [size_dist_ave[k] for k in Ls]
        S = np.zeros((len(size_dist_ave), max(map(len, size_dist_ave))))
        for i, s in enumerate(size_dist_ave):
            for j, num in enumerate(s):
                S[i][j] = num
        size_dist_ave = S
    else:
        size_dist_ave = []

    save_data.save("../results/data/diecutting/beta=%2.2f_" % beta,
                   beta=beta, num_of_strings=num_of_strings,
                   L=params['L'], frames=params['frames'],
                   Ls=Ls, N_sub=N_sub, size_dist=size_dist,
                   size_dist_ave=size_dist_ave)


if __name__ == '__main__':
    num_of_strings = 100

    parser = argparse.ArgumentParser()
    parser.add_argument('beta', type=float, nargs=1,
                        help='parameter beta (inverse temparature)')
    args = parser.parse_args()
    beta = args.beta[0]

    print "beta = %2.2f" % beta
    eval_simulation_for_one_beta(beta, num_of_strings)
