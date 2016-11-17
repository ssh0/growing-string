#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-11-17


from box_counting import main
import argparse


if __name__ == '__main__':

    sample = 8

    parser = argparse.ArgumentParser()
    parser.add_argument('beta', type=float, nargs=1,
                        help='parameter beta (inverse temparature)')
    args = parser.parse_args()
    for i in range(sample):
        print "====== loop: {}/{} ======".format(i + 1, sample)
        main(args.beta[0], plot=False)
