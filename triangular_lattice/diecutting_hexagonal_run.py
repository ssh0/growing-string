#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-10-16


from diecutting_hexagonal import DieCuttingHexagonal
import numpy as np
from tqdm import tqdm
import time


result_set = {
    'num_of_sub_clusters': {
        '_func': len,
        'func': np.average
    },
    # 'size_dist_of_sub_clusters': {
    #     '_func': lambda arr: np.bincount(map(len, arr)),
    #     'func': lambda arr: map(sum, itertools.izip_longest(*arr, fillvalue=0))
    # },
    # 'max_size_of_sub_cluster': {
    #     '_func': lambda arr: max(map(len, arr)),
    #     'func': np.average
    # }
}

def eval_simulation_for_each_string(params):
    main = DieCuttingHexagonal(params)
    main.start(result_set, visualize=False)
    return main.cutting_sizes, main.res['num_of_sub_clusters']

def eval_simulation_for_one_beta(beta, num_of_strings=30):
    current_time = time.strftime("%y%m%d_%H%M%S")
    params = {
        'L': 1000,
        'frames': 500,
        'beta': beta,
        'plot': False
    }

    d = {}
    for s in tqdm(range(num_of_strings)):
        _Ls, _N_sub = eval_simulation_for_each_string(params)
        for i, l in enumerate(_Ls):
            if d.has_key(l):
                d[l].append(_N_sub[i])
            else:
                d[l] = [_N_sub[i], ]

    mean = [(k, np.average(v)) for k, v in d.items()]
    Ls, N_sub = np.array(sorted(mean)).T

    result_data_path = "results/data/diecutting/beta=%2.2f" % beta
    result_data_path += "_" + current_time
    result_data_path += ".npz"
    np.savez(result_data_path,
            beta=beta,
            num_of_strings=num_of_strings,
            L=params['L'],
            frames=params['frames'],
            Ls=Ls,
            N_sub=N_sub
            )
    print "[saved] {}".format(result_data_path)

if __name__ == '__main__':
    # === Averaging (sample N: num_of_strings) ===
    beta = 6.
    num_of_strings = 30
    print "beta = %2.2f" % beta
    eval_simulation_for_one_beta(beta, num_of_strings)

