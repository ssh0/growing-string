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
    params = {
        'L': 1000,
        'frames': 500,
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
            size_dist[k] = map(sum, itertools.izip_longest(*v, fillvalue=0))
        Ls = sorted(size_dist.keys())
        size_dist = [size_dist[k] for k in sorted(size_dist.keys())]
        S = np.zeros((len(size_dist), max(map(len, size_dist))))
        for i, s in enumerate(size_dist):
            for j, num in enumerate(s):
                S[i][j] = num
        size_dist = S
    else:
        size_dist = []

    result_data_path = "results/data/diecutting/beta=%2.2f" % beta
    result_data_path += "_" + current_time
    result_data_path += ".npz"
    np.savez(result_data_path,
            beta=beta,
            num_of_strings=num_of_strings,
            L=params['L'],
            frames=params['frames'],
            Ls=Ls,
            N_sub=N_sub,
            size_dist=size_dist
            )
    print "[saved] {}".format(result_data_path)

if __name__ == '__main__':
    # === Averaging (sample N: num_of_strings) ===
    beta = 6.
    num_of_strings = 100
    print "beta = %2.2f" % beta
    eval_simulation_for_one_beta(beta, num_of_strings)

