import os
import matplotlib.pyplot as plt
import pickle
from itertools import chain
import math

from CES.desNet import Des

import numpy as np
from CES.ces import *
import cv2
import torch

from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr, kendalltau

import argparse


class Test4Category:
    def __init__(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.desnet = Des(device=device)  # UTR

    def test_4_cata_and_save_points(self, root_dir, seq_length, mode='CES', z_fill=3, ver='default'):
        print('Current Mode: ', mode)
        ref_dir = root_dir + '/ref/'
        points = []
        for cata in ['/CA/', '/SA/', '/PA/', '/WA/']:
            tgt_dir = root_dir + cata
            if ver == '0516':
                ref_dir = root_dir + cata + 'ref/'
                tgt_dir = root_dir + cata + 'mov/'
                z_fill = 4
            pt = []
            for idx in range(0, seq_length, 1):
                img_r = cv2.imread(ref_dir + str(idx).zfill(z_fill) + '.png', 0)
                img_m = cv2.imread(tgt_dir + str(idx).zfill(z_fill) + '.png', 0)
                sim = self.compute_similarity(img_r, img_m, mode=mode)
                pt.append(sim)
            # self.flier_counter(pt)
            points.append(pt)

        # save points
        with open(root_dir + '/points_' + mode + '.pkl', 'wb') as f:
            pickle.dump(points, f)


    @staticmethod
    def flier_counter(data):
        median = np.median(data)
        mean = np.mean(data)
        print('median {}, mean {}'.format(median, mean))
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        Up_lim = Q3 + 1.5 * IQR
        Down_lim = Q1 - 1.5 * IQR
        cnt = np.sum(data < Down_lim) + np.sum(data > Up_lim)
        # print('filer num {}'.format(cnt))
        print('OR: {}%'.format(cnt/len(data)*100))
        

    def compute_similarity(self, img_r, img_m, mode='CES'):
        sim = 0
        if mode == 'CES':
            sim = compute_CES_per_patch_pair(self.desnet, img_r, img_m, step=64, kp_step=8, mode='Mean')
        return sim

    # compute PLCC, SRCC in this function
    @staticmethod
    def compute_CC(func, popt, flatten_points, flatten_dist):
        pred_dist = func(flatten_points, *popt)
        # calculate PLCC and p-value
        plcc, _ = pearsonr(flatten_dist, pred_dist)
        # print results
        print("PLCC: ", plcc)
        # calculate SRCC and p-value
        srcc, _ = spearmanr(flatten_dist, pred_dist)
        print("SRCC: ", srcc)
        # calculate KRCC and p-value
        krcc, _ = kendalltau(flatten_dist, pred_dist)
        print("KRCC: ", krcc)

    def plot_figures_inverse(self, root_dir, points, mode,
                     fig='dist_plot', fit=True, discard_CA=False, p0=np.array([1, 1, 1, 1])):
        font1 = {'weight': 'bold',
                 'style': 'normal',
                 'size': 12,
                 }

        if fig == 'dist_plot':
            dist_lst = []
            fig, ax = plt.subplots()
            i = 0
            for cata in ['CA', 'SA', 'PA', 'WA']:
                with open(root_dir + '/{}.pkl'.format(cata), 'rb') as f:
                    dist_i = pickle.load(f)
                ax.scatter(dist_i, points[i], s=1, label=cata)
                # self.flier_counter(points[i])
                dist_lst.append(dist_i)
                i += 1

            # Fit the data to defined function
            if discard_CA:
                flatten_dist = np.array(list(chain(*dist_lst)))[len(dist_i):]
                flatten_points = np.array(list(chain(*points)))[len(dist_i):]
            else:
                flatten_dist = np.array(list(chain(*dist_lst)))
                flatten_points = np.array(list(chain(*points)))

            if fit:  # compute fit function
                def func(x, eta1, eta2, eta3, eta4):  # fitting curve from DISTS, suitable CES
                    return (eta1 - eta2) / (1 + np.exp(-(x - eta3) / np.abs(eta4))) + eta2

                if (p0 == np.array([1, 1, 1, 1])).all():
                    popt, pcov = curve_fit(func, flatten_dist, flatten_points)
                else:
                    popt = p0
                print('estimated params:{}'.format(popt))

            points = np.linspace(0, 20, 500)

            if fit:  # draw fit curve
                # print CC status
                self.compute_CC(func, popt, flatten_dist, flatten_points)
                # plot fitting curve
                plt.plot(points, func(points, *popt), 'r-', label='Fitted Curve')

            plt.xlim((-1, 21))
            plt.ylabel(mode, font1)
            plt.xlabel('Translation Distance, px', font1)
            plt.legend()
            os.makedirs(root_dir + '/dist_figs_inverse/', exist_ok=True)
            plt.savefig(root_dir + '/dist_figs_inverse/' + mode + '_distplot.tif')
            plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default='./data/EXP2_FlyEM_BS/Dataset_z32nm')
    parser.add_argument("--seq_length", type=int, default=640)
    args = parser.parse_args()
    root_dir = args.root_dir
    seq_length = args.seq_length
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    T4C = Test4Category()

    mode = 'CES'
    # compute and save points
    T4C.test_4_cata_and_save_points(root_dir, seq_length=seq_length, mode=mode)
    # draw scatter plots and fitting curve
    T4C.plot_figures_inverse(root_dir, points, mode, fig='dist_plot', fit=True, discard_CA=False,
                             p0=np.array([1, 1, 1, 1]))
