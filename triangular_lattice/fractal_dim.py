#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-12-01

import numpy as np
import matplotlib.pyplot as plt
from optimize import Optimize_powerlaw


def calc_tortuosity_for_each_beta(ax, i, filename):
    data = np.load(filename)
    beta = data['beta']
    num_of_strings = data['num_of_strings']
    L = data['L']
    frames = data['frames']
    distance_list = data['distance_list']
    path_length = data['path_length']

    num_of_pairs = 300

    x = np.array(path_length)
    x = x.reshape((num_of_strings, int(x.shape[0] / num_of_strings / num_of_pairs), num_of_pairs))

    y = np.array(distance_list)
    y = y.reshape((num_of_strings, int(y.shape[0] / num_of_strings / num_of_pairs), num_of_pairs))
    # tortuosity = length_of_the_curve / distance_between_the_ends_of_it
    y_ave = np.average(y, axis=2)

    X = np.average(y_ave, axis=0)
    Y = x[0, :, 0]
    ax.loglog(X, Y, ls='', marker='.', label=r'$\beta = %2.2f$' % beta,
              alpha=0.5)

    optimizer = Optimize_powerlaw(args=(X[:], Y[:]), parameters=[0.1, 2.])
    result = optimizer.fitting()
    print "beta = {}, D = {}".format(beta, result['D'])
    ax.loglog(X, optimizer.fitted(X), ls='-', marker='', color='k',
              label=r'$D = %2.2f$' % result['D'])
    return beta, result['D']

if __name__ == '__main__':
    import time

    result_data_path = {
        '0': "./results/data/distances/beta=0.00_161012_171430.npz",
        '5': "./results/data/distances/beta=5.00_161012_171649.npz",
        '10': "./results/data/distances/beta=10.00_161012_172119.npz",
        '15': "./results/data/distances/beta=15.00_161012_172209.npz",
        '20': "./results/data/distances/beta=20.00_161012_172338.npz",
        '0-2': "./results/data/distances/beta=0.00_161015_153311.npz",
        '5-2': "./results/data/distances/beta=5.00_161015_153838.npz",
        '10-2': "./results/data/distances/beta=10.00_161015_154048.npz",
        '15-2': "./results/data/distances/beta=15.00_161015_154136.npz",
        '20-2': "./results/data/distances/beta=20.00_161015_154419.npz",
        "0-3": "./results/data/distances/beta=0.00_161018_160133.npz",
        "1-3": "./results/data/distances/beta=1.00_161018_161118.npz",
        "2-3": "./results/data/distances/beta=2.00_161018_162849.npz",
        "3-3": "./results/data/distances/beta=3.00_161018_164500.npz",
        "4-3": "./results/data/distances/beta=4.00_161018_170824.npz",
        "5-3": "./results/data/distances/beta=5.00_161018_172135.npz",
        "6-3": "./results/data/distances/beta=6.00_161018_173918.npz",
        "7-3": "./results/data/distances/beta=7.00_161018_175342.npz",
        "8-3": "./results/data/distances/beta=8.00_161018_180914.npz",
        "9-3": "./results/data/distances/beta=9.00_161018_182543.npz",
        "10-3": "./results/data/distances/beta=10.00_161018_184136.npz",
        "0-4": "./results/data/distances/beta=0.00_161018_202019.npz",
        "5-4": "./results/data/distances/beta=5.00_161018_202404.npz",
        "10-4": "./results/data/distances/beta=10.00_161018_202616.npz",
        "15-4": "./results/data/distances/beta=15.00_161018_202846.npz",
        "20-4": "./results/data/distances/beta=20.00_161018_203002.npz",
    }

    distance_data = {
        # 120: [
        #     "./results/data/distances/frames=120_beta=0.00_161208_143218.npz",
        #     "./results/data/distances/frames=120_beta=2.00_161208_143223.npz",
        #     "./results/data/distances/frames=120_beta=4.00_161208_143232.npz",
        #     "./results/data/distances/frames=120_beta=6.00_161208_143301.npz",
        #     "./results/data/distances/frames=120_beta=8.00_161208_143310.npz",
        #     "./results/data/distances/frames=120_beta=10.00_161208_143329.npz",
        # ],
        150: [
            "./results/data/distances/frames=150_beta=0.00_161208_210328.npz",
            "./results/data/distances/frames=150_beta=10.00_161208_210407.npz",
            "./results/data/distances/frames=150_beta=2.00_161208_210345.npz",
            "./results/data/distances/frames=150_beta=4.00_161208_210350.npz",
            "./results/data/distances/frames=150_beta=6.00_161208_210354.npz",
            "./results/data/distances/frames=150_beta=8.00_161208_210400.npz",
        ],
        300: [
            "./results/data/distances/frames=300_beta=0.00_161208_143342.npz",
            "./results/data/distances/frames=300_beta=2.00_161208_143424.npz",
            "./results/data/distances/frames=300_beta=4.00_161208_143422.npz",
            "./results/data/distances/frames=300_beta=6.00_161208_143555.npz",
            "./results/data/distances/frames=300_beta=8.00_161208_143542.npz",
            "./results/data/distances/frames=300_beta=10.00_161208_143607.npz",
        ],
        450: [
            "./results/data/distances/frames=450_beta=0.00_161208_143759.npz",
            "./results/data/distances/frames=450_beta=2.00_161208_143909.npz",
            "./results/data/distances/frames=450_beta=4.00_161208_143907.npz",
            "./results/data/distances/frames=450_beta=6.00_161208_144125.npz",
            "./results/data/distances/frames=450_beta=8.00_161208_144056.npz",
            "./results/data/distances/frames=450_beta=10.00_161208_144230.npz",
        ],
        600: [
            "./results/data/distances/frames=600_beta=0.00_161208_144639.npz",
            "./results/data/distances/frames=600_beta=2.00_161208_144652.npz",
            "./results/data/distances/frames=600_beta=4.00_161208_144856.npz",
            "./results/data/distances/frames=600_beta=6.00_161208_145059.npz",
            "./results/data/distances/frames=600_beta=8.00_161208_145203.npz",
            "./results/data/distances/frames=600_beta=10.00_161208_145325.npz",
        ],
        750: [
            "./results/data/distances/frames=750_beta=0.00_161208_145806.npz",
            "./results/data/distances/frames=750_beta=2.00_161208_150017.npz",
            "./results/data/distances/frames=750_beta=4.00_161208_150402.npz",
            "./results/data/distances/frames=750_beta=6.00_161208_150608.npz",
            "./results/data/distances/frames=750_beta=8.00_161208_150936.npz",
            "./results/data/distances/frames=750_beta=10.00_161208_150951.npz",
        ],
        900: [
            "./results/data/distances/frames=900_beta=0.00_161208_151703.npz",
            "./results/data/distances/frames=900_beta=2.00_161208_152028.npz",
            "./results/data/distances/frames=900_beta=4.00_161208_152651.npz",
            "./results/data/distances/frames=900_beta=6.00_161208_152859.npz",
            "./results/data/distances/frames=900_beta=8.00_161208_153233.npz",
            "./results/data/distances/frames=900_beta=10.00_161208_153519.npz",
        ],
        1050: [
            "./results/data/distances/frames=1050_beta=0.00_161208_154248.npz",
            "./results/data/distances/frames=1050_beta=2.00_161208_154817.npz",
            "./results/data/distances/frames=1050_beta=4.00_161208_155631.npz",
            "./results/data/distances/frames=1050_beta=6.00_161208_160054.npz",
            "./results/data/distances/frames=1050_beta=8.00_161208_160730.npz",
            "./results/data/distances/frames=1050_beta=10.00_161208_160946.npz",
        ],
        1200: [
            "./results/data/distances/frames=1200_beta=0.00_161208_161554.npz",
            "./results/data/distances/frames=1200_beta=2.00_161208_162259.npz",
            "./results/data/distances/frames=1200_beta=4.00_161208_162955.npz",
            "./results/data/distances/frames=1200_beta=6.00_161208_163345.npz",
            "./results/data/distances/frames=1200_beta=8.00_161208_163842.npz",
            "./results/data/distances/frames=1200_beta=10.00_161208_164024.npz",
        ]
    }


    def fractal_dims():
        fig, (ax1, ax2) = plt.subplots(2, 1)
        betas, Ds = [], []

        # for i in [0, 5, 10, 15, 20]:
        # for i in [0]:
        for i in [0, 2, 4, 8, 10]:
        # for i in range(11):
            # fn = result_data_path['%d-2' % i]
            fn = result_data_path['%d-3' % i]
            # fn = result_data_path['%d-4' % i]
            beta, D = calc_tortuosity_for_each_beta(ax1, i, fn)
            betas.append(beta)
            Ds.append(D)

        ax1.set_ylim(0., ax1.get_ylim()[1])
        ax1.set_xlabel(r'Averaged distance $\lambda_{\mathrm{avg}}$')
        ax1.set_ylabel(r'Path length $L$')
        ax1.set_title(r'$\lambda_{\mathrm{avg}}$ -- $L$')
        ax1.legend(loc='best')

        ax2.set_xlabel(r'$\beta$')
        ax2.set_ylabel(r'Fractal dimension $D$')
        ax2.set_title(r'$\beta$ -- $D$')
        ax2.plot(betas, Ds, 'o')

        plt.show()

    def variation_of_fd(plot=True, save_image=False):
        T = []
        D_T = []
        for frames in sorted(distance_data.keys()):
            T.append(frames)
            fig, ax1 = plt.subplots()
            betas, Ds = [], []
            for i, f in enumerate(distance_data[frames]):
                beta, D = calc_tortuosity_for_each_beta(ax1, i, f)
                betas.append(beta)
                Ds.append(D)

            D_T.append(Ds)

            ax1.set_ylim(0., ax1.get_ylim()[1])
            ax1.set_xlabel(r'Averaged distance $\lambda_{\mathrm{avg}}$')
            ax1.set_ylabel(r'Path length $L$')
            ax1.set_title(r'$\lambda_{\mathrm{avg}}$ -- $L$ (frames=%d)' % frames)
            ax1.legend(loc='best')

            if plot:
                plt.show()
            elif save_image:
                result_image_path = "results/img/fractal_dim/d_L_frames=%d" % frames
                result_image_path += "_" + time.strftime("%y%m%d_%H%M%S")
                result_image_path += ".png"
                plt.savefig(result_image_path)
                plt.close()
                print "[saved] " + result_image_path
            else:
                plt.close()

            fig, ax2 = plt.subplots()
            ax2.set_xlabel(r'$\beta$')
            ax2.set_ylabel(r'Fractal dimension $D$')
            ax2.set_title(r'$\beta$ -- $D$')
            ax2.plot(betas, Ds, 'o')

            if plot:
                plt.show()
            elif save_image:
                result_image_path = "results/img/fractal_dim/t_D_frames=%d" % frames
                result_image_path += "_" + time.strftime("%y%m%d_%H%M%S")
                result_image_path += ".png"
                plt.savefig(result_image_path)
                plt.close()
                print "[saved] " + result_image_path
            else:
                plt.close()

        T = np.array(T)
        D = np.array(D_T).T
        betas = np.array(betas)
        fig, ax = plt.subplots()
        for beta, D_beta in zip(betas, D):
            ax.plot(T, D_beta, ls='', marker='.',
                    label=r'$\beta = %2.2f$' % beta,
                    )

        ax.set_xlim(0, ax.get_xlim()[1] + 100)
        ax.set_ylim(1., ax.get_ylim()[1])
        ax.set_xlabel(r'$T$')
        ax.set_ylabel(r'$D(T)$')
        ax.set_title('Fractal dimension')
        ax.legend(loc='best')
        plt.show()

    # fractal_dims()
    variation_of_fd(plot=False, save_image=True)
