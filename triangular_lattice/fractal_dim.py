#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-12-01

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d.axes3d import Axes3D
from optimize import Optimize_powerlaw


def calc_fractal_dim_for_each_beta(ax, i, filename):
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
            "./results/data/distances/frames=150_beta=2.00_161208_210345.npz",
            "./results/data/distances/frames=150_beta=4.00_161208_210350.npz",
            "./results/data/distances/frames=150_beta=6.00_161208_210354.npz",
            "./results/data/distances/frames=150_beta=8.00_161208_210400.npz",
            "./results/data/distances/frames=150_beta=10.00_161208_210407.npz",
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
        ],
        1350: [
            "./results/data/distances/frames=1350_beta=0.00_161219_163059.npz",
            "./results/data/distances/frames=1350_beta=2.00_161219_163229.npz",
            "./results/data/distances/frames=1350_beta=4.00_161219_200509.npz",
            "./results/data/distances/frames=1350_beta=6.00_161219_230504.npz",
            "./results/data/distances/frames=1350_beta=8.00_161220_020451.npz",
            "./results/data/distances/frames=1350_beta=10.00_161220_050348.npz",
        ],
        1500: [
            "./results/data/distances/frames=1500_beta=0.00_161219_170034.npz",
            "./results/data/distances/frames=1500_beta=2.00_161219_170406.npz",
            "./results/data/distances/frames=1500_beta=4.00_161219_203520.npz",
            "./results/data/distances/frames=1500_beta=6.00_161219_233306.npz",
            "./results/data/distances/frames=1500_beta=8.00_161220_023318.npz",
            "./results/data/distances/frames=1500_beta=10.00_161220_053252.npz",
        ],
        1650: [
            "./results/data/distances/frames=1650_beta=0.00_161219_173703.npz",
            "./results/data/distances/frames=1650_beta=2.00_161219_174118.npz",
            "./results/data/distances/frames=1650_beta=4.00_161219_211624.npz",
            "./results/data/distances/frames=1650_beta=6.00_161220_000711.npz",
            "./results/data/distances/frames=1650_beta=8.00_161220_030835.npz",
            "./results/data/distances/frames=1650_beta=10.00_161220_061122.npz",
        ],
        1800: [
            "./results/data/distances/frames=1800_beta=0.00_161219_181458.npz",
            "./results/data/distances/frames=1800_beta=2.00_161219_182205.npz",
            "./results/data/distances/frames=1800_beta=4.00_161219_215624.npz",
            "./results/data/distances/frames=1800_beta=6.00_161220_005219.npz",
            "./results/data/distances/frames=1800_beta=8.00_161220_035046.npz",
            "./results/data/distances/frames=1800_beta=10.00_161220_065509.npz",
        ],
        1950: [
            "./results/data/distances/frames=1950_beta=0.00_161219_185939.npz",
            "./results/data/distances/frames=1950_beta=2.00_161219_190733.npz",
            "./results/data/distances/frames=1950_beta=4.00_161219_224207.npz",
            "./results/data/distances/frames=1950_beta=6.00_161220_014202.npz",
            "./results/data/distances/frames=1950_beta=8.00_161220_044049.npz",
            "./results/data/distances/frames=1950_beta=10.00_161220_074620.npz",
        ],

        # 150: [
        #     "./results/data/distances/frames=150_beta=0.00_161220_144021.npz",
        #     "./results/data/distances/frames=150_beta=2.00_161220_174953.npz",
        #     "./results/data/distances/frames=150_beta=4.00_161220_212543.npz",
        #     "./results/data/distances/frames=150_beta=6.00_161221_013534.npz",
        #     "./results/data/distances/frames=150_beta=8.00_161221_053006.npz",
        #     "./results/data/distances/frames=150_beta=10.00_161221_092909.npz",
        # ],
        # 300: [
        #     "./results/data/distances/frames=300_beta=0.00_161220_144115.npz",
        #     "./results/data/distances/frames=300_beta=2.00_161220_175052.npz",
        #     "./results/data/distances/frames=300_beta=4.00_161220_212654.npz",
        #     "./results/data/distances/frames=300_beta=6.00_161221_013639.npz",
        #     "./results/data/distances/frames=300_beta=8.00_161221_053108.npz",
        #     "./results/data/distances/frames=300_beta=10.00_161221_093013.npz",
        # ],
        # 450: [
        #     "./results/data/distances/frames=450_beta=0.00_161220_144320.npz",
        #     "./results/data/distances/frames=450_beta=2.00_161220_175307.npz",
        #     "./results/data/distances/frames=450_beta=4.00_161220_212927.npz",
        #     "./results/data/distances/frames=450_beta=6.00_161221_013901.npz",
        #     "./results/data/distances/frames=450_beta=8.00_161221_053332.npz",
        #     "./results/data/distances/frames=450_beta=10.00_161221_093236.npz",
        # ],
        # 600: [
        #     "./results/data/distances/frames=600_beta=0.00_161220_144707.npz",
        #     "./results/data/distances/frames=600_beta=2.00_161220_175701.npz",
        #     "./results/data/distances/frames=600_beta=4.00_161220_213353.npz",
        #     "./results/data/distances/frames=600_beta=6.00_161221_014314.npz",
        #     "./results/data/distances/frames=600_beta=8.00_161221_053748.npz",
        #     "./results/data/distances/frames=600_beta=10.00_161221_093657.npz",
        # ],
        # 750: [
        #     "./results/data/distances/frames=750_beta=0.00_161220_145258.npz",
        #     "./results/data/distances/frames=750_beta=2.00_161220_180302.npz",
        #     "./results/data/distances/frames=750_beta=4.00_161220_214108.npz",
        #     "./results/data/distances/frames=750_beta=6.00_161221_015024.npz",
        #     "./results/data/distances/frames=750_beta=8.00_161221_054453.npz",
        #     "./results/data/distances/frames=750_beta=10.00_161221_094356.npz",
        # ],
        # 900: [
        #     "./results/data/distances/frames=900_beta=0.00_161220_150120.npz",
        #     "./results/data/distances/frames=900_beta=2.00_161220_181200.npz",
        #     "./results/data/distances/frames=900_beta=4.00_161220_215230.npz",
        #     "./results/data/distances/frames=900_beta=6.00_161221_020013.npz",
        #     "./results/data/distances/frames=900_beta=8.00_161221_055443.npz",
        #     "./results/data/distances/frames=900_beta=10.00_161221_095400.npz",
        # ],
        # 1050: [
        #     "./results/data/distances/frames=1050_beta=0.00_161220_151234.npz",
        #     "./results/data/distances/frames=1050_beta=2.00_161220_182408.npz",
        #     "./results/data/distances/frames=1050_beta=4.00_161220_220804.npz",
        #     "./results/data/distances/frames=1050_beta=6.00_161221_021404.npz",
        #     "./results/data/distances/frames=1050_beta=8.00_161221_060827.npz",
        #     "./results/data/distances/frames=1050_beta=10.00_161221_100741.npz",
        # ],
        # 1200: [
        #     "./results/data/distances/frames=1200_beta=0.00_161220_152914.npz",
        #     "./results/data/distances/frames=1200_beta=2.00_161220_183941.npz",
        #     "./results/data/distances/frames=1200_beta=4.00_161220_222724.npz",
        #     "./results/data/distances/frames=1200_beta=6.00_161221_023130.npz",
        #     "./results/data/distances/frames=1200_beta=8.00_161221_062652.npz",
        #     "./results/data/distances/frames=1200_beta=10.00_161221_102628.npz",
        # ],
        # 1350: [
        #     "./results/data/distances/frames=1350_beta=0.00_161220_154801.npz",
        #     "./results/data/distances/frames=1350_beta=2.00_161220_190107.npz",
        #     "./results/data/distances/frames=1350_beta=4.00_161220_224925.npz",
        #     "./results/data/distances/frames=1350_beta=6.00_161221_025428.npz",
        #     "./results/data/distances/frames=1350_beta=8.00_161221_065034.npz",
        #     "./results/data/distances/frames=1350_beta=10.00_161221_105035.npz",
        # ],
        # 1500: [
        #     "./results/data/distances/frames=1500_beta=0.00_161220_161154.npz",
        #     "./results/data/distances/frames=1500_beta=2.00_161220_192658.npz",
        #     "./results/data/distances/frames=1500_beta=4.00_161220_232018.npz",
        #     "./results/data/distances/frames=1500_beta=6.00_161221_032233.npz",
        #     "./results/data/distances/frames=1500_beta=8.00_161221_071933.npz",
        #     "./results/data/distances/frames=1500_beta=10.00_161221_112018.npz",
        # ],
        # 1650: [
        #     "./results/data/distances/frames=1650_beta=0.00_161220_163934.npz",
        #     "./results/data/distances/frames=1650_beta=2.00_161220_195922.npz",
        #     "./results/data/distances/frames=1650_beta=4.00_161221_000037.npz",
        #     "./results/data/distances/frames=1650_beta=6.00_161221_035826.npz",
        #     "./results/data/distances/frames=1650_beta=8.00_161221_075528.npz",
        #     "./results/data/distances/frames=1650_beta=10.00_161221_115541.npz",
        # ],
        # 1800: [
        #     "./results/data/distances/frames=1800_beta=0.00_161220_171202.npz",
        #     "./results/data/distances/frames=1800_beta=2.00_161220_203633.npz",
        #     "./results/data/distances/frames=1800_beta=4.00_161221_004752.npz",
        #     "./results/data/distances/frames=1800_beta=6.00_161221_044012.npz",
        #     "./results/data/distances/frames=1800_beta=8.00_161221_083736.npz",
        #     "./results/data/distances/frames=1800_beta=10.00_161221_123947.npz",
        # ],
        # 1950: [
        #     "./results/data/distances/frames=1950_beta=0.00_161220_174934.npz",
        #     "./results/data/distances/frames=1950_beta=2.00_161220_212524.npz",
        #     "./results/data/distances/frames=1950_beta=4.00_161221_013517.npz",
        #     "./results/data/distances/frames=1950_beta=6.00_161221_052949.npz",
        #     "./results/data/distances/frames=1950_beta=8.00_161221_092852.npz",
        #     "./results/data/distances/frames=1950_beta=10.00_161221_133606.npz",
        # ]
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
            beta, D = calc_fractal_dim_for_each_beta(ax1, i, fn)
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
                beta, D = calc_fractal_dim_for_each_beta(ax1, i, f)
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

        # X, Y = np.meshgrid(betas, T)
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')

        # ax.plot_wireframe(X, Y, D.T, rstride=1)
        # ax.set_zlim(1., ax.get_zlim()[1])
        # ax.set_title('Fractal dimension')
        # ax.set_xlabel(r'$\beta$')
        # ax.set_ylabel(r'$T$')
        # ax.set_zlabel(r'$D(T)$')
        # plt.show()

        fig, ax = plt.subplots()
        for i, (beta, D_beta) in enumerate(zip(betas, D)):
            color = cm.viridis(float(i) / (len(betas) - 1))
            ax.plot(T, D_beta, ls='', marker='o',
                    label=r'$\beta = %2.2f$' % beta,
                    color=color,
                    markeredgewidth=0.0
                    )

        ax.set_xlim(0, ax.get_xlim()[1] + 100)
        ax.set_ylim(1., ax.get_ylim()[1])
        ax.set_xlabel(r'$T$')
        ax.set_ylabel(r'$D(T)$')
        ax.set_title('Fractal dimension')
        ax.legend(loc='best')
        plt.show()

    # fractal_dims()
    variation_of_fd(plot=False, save_image=False)
