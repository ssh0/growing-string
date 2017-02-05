#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-12-06

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import gamma
import time
import set_data_path


def load_data(_path):
    data = np.load(_path)
    beta = data['beta']
    num_of_strings = data['num_of_strings']
    frames = data['frames']
    Ls = data['Ls'].astype(np.float)
    # Ls = (3 * Ls * (Ls + 1) + 1)
    size_dist = data['size_dist']

    # M = np.array([np.sum(l) for l in size_dist]) - 1.
    M = np.array([np.sum(l) - 1 for l in size_dist])
    M_ave = M / np.sum(M)

    return {
        'beta': beta,
        'num_of_strings': num_of_strings,
        'frames': frames,
        'Ls': Ls,
        'M_ave': M_ave
    }

def show_plot1(ax, num_of_strings):
    ax.legend(loc='best')
    ax.set_ylim((0., ax.get_ylim()[1]))
    ax.set_title('Strings in hexagonal region' +
                ' (sample: {})'.format(num_of_strings))
    ax.set_xlabel(r'Cutting size $L$')
    ax.set_ylabel('Average number of the sub-clusters (normalized)')

def fit_a_x0_scale(path):
    betas = []
    a = []
    loc = []
    scale = []

    fig, ax = plt.subplots()
    for i, result_data_path in enumerate(path):
        globals().update(load_data(result_data_path))

        ax.plot(Ls, M_ave, '.', label=r'$\beta = %2.2f$' % beta,
                color=cm.viridis(float(i) / len(path)))

        popt = curve_fit(gamma.pdf, xdata=Ls, ydata=M_ave, p0=[2.5, -5., 10.])[0]
        print beta, popt
        betas.append(beta)

        a.append(popt[0])
        loc.append(popt[1])
        scale.append(popt[2])

        x = np.linspace(0, max(Ls), num=5*max(Ls))
        ax.plot(x, gamma.pdf(x, a=popt[0], loc=popt[1], scale=popt[2]),
                    '-', label=r'fitted $\beta = %2.2f$' % beta,
                    color=cm.viridis(float(i) / len(path)))
    show_plot1(ax, num_of_strings)
    plt.show()

    betas = np.array(betas)
    a = np.array(a)
    loc = np.array(loc)
    scale = np.array(scale)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax1.plot(betas, a, 'o')
    [ax.set_xlabel(r'$\beta$') for ax in [ax1, ax2, ax3]]
    [ax.set_xlim((0, max(betas))) for ax in [ax1, ax2, ax3]]
    ax1.set_ylabel(r'Shape parameter: $a$')
    ax2.plot(betas, loc, 'o')
    ax2.set_ylabel(r'Translation parameter: $x_{0}$')
    # ax3.plot(-betas, -scale)  # お試し
    ax3.plot(betas, scale, 'o')
    ax3.set_ylabel(r'Scale parameter: $\theta$')
    plt.show()

def fit_a_scale(path, fixed_loc):

    def modified_gamma(x, a, scale):
        # loc = c * a + d
        loc = fixed_loc
        return gamma.pdf(x, a=a, loc=loc, scale=scale)

    betas = []
    a = []
    scale = []

    fig, ax = plt.subplots()
    for i, result_data_path in enumerate(path):
        globals().update(load_data(result_data_path))
        ax.plot(Ls, M_ave, '.', label=r'$\beta = %2.2f$' % beta,
                color=cm.viridis(float(i) / len(path)))
        popt = curve_fit(modified_gamma, xdata=Ls, ydata=M_ave, p0=[2.5, 10.])[0]
        print beta, popt
        betas.append(beta)

        a.append(popt[0])
        scale.append(popt[1])

        x = np.linspace(0, max(Ls), num=5*max(Ls))
        ax.plot(x, modified_gamma(x, a=popt[0], scale=popt[1]),
                    '-', label=r'fitted $\beta = %2.2f$' % beta,
                    color=cm.viridis(float(i) / len(path)))
    show_plot1(ax, num_of_strings)
    plt.show()

    betas = np.array(betas)
    a = np.array(a)
    scale = np.array(scale)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.set_title(r'Fitting parameter (fixed: $x_{0} = 0$)')
    ax1.plot(betas, a, 'o')
    [ax.set_xlabel(r'$\beta$') for ax in [ax1, ax2]]
    [ax.set_xlim((0, max(betas))) for ax in [ax1, ax2]]
    ax1.set_ylabel(r'Shape parameter: $a$')
    ax2.plot(betas, scale, 'o')
    ax2.set_ylabel(r'Scale parameter: $\theta$')
    plt.show()

def fit_scale(path, fixed_a, fixed_loc, save_image=False):

    matplotlib.rcParams['savefig.dpi'] = 300

    def modified_gamma_2(x, scale):
        a = fixed_a
        loc = fixed_loc
        return gamma.pdf(x, a=a, loc=loc, scale=scale)

    betas = []
    scale = []

    fig, ax = plt.subplots()
    for i, result_data_path in enumerate(path):
        globals().update(load_data(result_data_path))
        ax.plot(Ls, M_ave, '.', label=r'$\beta = %2.2f$' % beta,
                color=cm.viridis(float(i) / len(path)))
        popt = curve_fit(modified_gamma_2, xdata=Ls, ydata=M_ave, p0=[10.])[0]
        print beta, popt
        betas.append(beta)
        scale.append(popt[0])

        x = np.linspace(0, max(Ls), num=5*max(Ls))
        ax.plot(x, modified_gamma_2(x, scale=popt[0]),
                '-',
                # label=r'fitted $\beta = %2.2f$' % beta,
                color=cm.viridis(float(i) / len(path)))

        ## critical point
        # critcal_point = 2. * popt[0]  # x = (a - 1) * scale
        # ax.plot([critcal_point] * 2, [0., 0.05], '-',
        #         color=cm.viridis(float(i) / len(path)))
    show_plot1(ax, num_of_strings)

    if save_image:
        result_image_path = "../results/img/diecutting/fitted_gamma_fixed_a_x0"
        result_image_path += "_" + time.strftime("%y%m%d_%H%M%S")
        pdf = PdfPages(result_image_path + ".pdf")
        plt.savefig(result_image_path + ".png")
        pdf.savefig()
        pdf.close()
        plt.close()
        print "[saved] " + result_image_path
    else:
        plt.show()
        plt.close()

    betas = np.array(betas)
    scale = np.array(scale)

    # beta_theta = lambda x, a, b: a*x + b
    beta_theta = lambda x, a, b: a*np.log(x) + b

    fig, ax = plt.subplots()
    ax.set_title(r'Fitting parameter')
    ax.plot(betas, scale, 'o')
    popt = curve_fit(beta_theta, xdata=betas, ydata=scale, p0=[15., 0.])[0]
    x = np.linspace(min(betas), max(betas))
    # ax.plot(x, beta_theta(x, popt[0], popt[1]), '-', color='k',
    #         label=r'$\theta = {} \beta + {}$'.format(*popt),
    #         )
    ax.plot(x, beta_theta(x, popt[0], popt[1]), '-', color='k',
            label=r'$\theta = {} \log \beta + {}$'.format(*popt),
            )
    ax.legend(loc='best')
    ax.set_xlim((0, max(betas)))
    ax.set_ylim((0, ax.get_ylim()[1]))
    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel(r'Scale parameter: $\theta$')

    if save_image:
        result_image_path = "../results/img/diecutting/fitted_parameters_fixed_a_x0"
        result_image_path += "_" + time.strftime("%y%m%d_%H%M%S")
        pdf = PdfPages(result_image_path + ".pdf")
        plt.savefig(result_image_path + ".png")
        pdf.savefig()
        pdf.close()
        plt.close()
        print "[saved] " + result_image_path
    else:
        plt.show()
        plt.close()

    plt.show()

def no_fit(path, fixed_a, fixed_loc, _a, _b, save_image=False):

    matplotlib.rcParams['savefig.dpi'] = 300

    def modified_gamma_3(x, beta):
        a = fixed_a
        loc = fixed_loc
        # scale = _a * beta + _b
        scale = _a * np.log(beta) + _b
        return gamma.pdf(x, a=a, loc=loc, scale=scale)

    betas = []
    scale = []

    fig, ax = plt.subplots()
    for i, result_data_path in enumerate(path):
        globals().update(load_data(result_data_path))
        ax.plot(Ls, M_ave, '.', label=r'$\beta = %2.2f$' % beta,
                color=cm.viridis(float(i) / len(path)), alpha=0.5)
        betas.append(beta)

        x = np.linspace(0, max(Ls), num=5*max(Ls))
        ax.plot(x, modified_gamma_3(x, beta),
                '-',
                # label=r'fitted $\beta = %2.2f$' % beta,
                color=cm.viridis(float(i) / len(path)))

    show_plot1(ax, num_of_strings)

    if save_image:
        result_image_path = "../results/img/diecutting/fitted_gamma_fixed_a_x0"
        result_image_path += "_" + time.strftime("%y%m%d_%H%M%S")
        pdf = PdfPages(result_image_path + ".pdf")
        plt.savefig(result_image_path + ".png")
        pdf.savefig()
        pdf.close()
        plt.close()
        print "[saved] " + result_image_path
    else:
        plt.show()
        plt.close()


if __name__ == '__main__':
    # fit_a_x0_scale(set_data_path.data_path)
    # fit_a_scale(set_data_path.data_path, fixed_loc=0.)
    # fit_scale(set_data_path.data_path, fixed_a=3., fixed_loc=0., save_image=False)
    # no_fit(set_data_path.data_path, fixed_a=3., fixed_loc=0., _a=3.6, _b=0.,  save_image=False)
    no_fit(set_data_path.data_path, fixed_a=3., fixed_loc=0., _a=19., _b=-8.,  save_image=False)
