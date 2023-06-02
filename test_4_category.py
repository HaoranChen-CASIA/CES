import os
import matplotlib.pyplot as plt
import pickle
from itertools import chain
import math

from CES.desNet import Des
from NCCnet.test import load_model
from Judge.patch_similarity import *
from MiRASuperPoint.superpoint import SuperPoint
from HardNet.extract_hardnet_desc_from_hpatches_file import *
from AANet.export_descriptor_sift import *
from DISTS_pytorch.DISTS_pt import DISTS

from skimage.metrics import structural_similarity as ssim
import numpy as np
from CES.ces import *
import cv2
import torch
from sklearn.metrics import mean_squared_error

from scipy.optimize import curve_fit, root
from scipy.stats import pearsonr, spearmanr, kendalltau

from skimage.metrics import normalized_mutual_information


class Test4Category:
    def __init__(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.desnet = Des(device=device)  # UTR
        self.nccnet = load_model('../NCCnet/')  # NCC Net, trained on fib-mito patches
        self.sift = cv2.SIFT_create()  # sift descriptor
        self.orb = cv2.ORB_create()  # BRIEF descriptor
        self.superpoint = SuperPoint(device=device)  # superpoint descriptor
        self.hardnet = load_hardnet('../HardNet/HardNet++.pth', DO_CUDA=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.aanet = load_AAnet(device, '../AANet/aanet.pth')
        self.dists = DISTS(weights_path='../DISTS_pytorch/weights.pt').to(device)

    def test_4_cata_and_save_points(self, root_dir, seq_length, mode='CES', z_fill=3, ver='default'):
        print('Current Mode: ', mode)
        ref_dir = root_dir + '/ref/'
        points = []
        if mode == 'SIFT+l2dist' or mode == 'denseSIFT+l2dist':
            desc_method = self.sift
        else:
            desc_method = self.orb
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
                sim = self.compute_similarity(desc_method, img_r, img_m, mode=mode)
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

    def plot_figures(self, root_dir, points, mode,
                     fig='dist_plot', fit=True, discard_CA=False, p0=np.array([1, 1, 1, 1])):
        font1 = {'weight': 'bold',
                 'style': 'normal',
                 'size': 12,
                 }
        if fig == 'box_plot':
            fig, ax = plt.subplots()
            ax.boxplot(points, showmeans=True)
            ax.set_xticklabels(['CA', 'SA', 'PA', 'WA'])

            plt.ylim((-0.2, 1))
            if mode == 'ORB+hamming':
                plt.ylim((0, 200))
            # if mode == 'denseSuperPoint+l2dist' or mode == 'denseUTR+l2dist' or 'denseHardNet+l2dist':
            if mode.find('l2dist') != -1:
                plt.ylim((0, 2))
            plt.xlabel('Align Quality', font1)
            plt.ylabel(mode, font1)
            plt.savefig(root_dir + '/dist_figs/' + mode + '_boxplot.tif')

        if fig == 'dist_plot':
            dist_lst = []
            fig, ax = plt.subplots()
            i = 0
            for cata in ['CA', 'SA', 'PA', 'WA']:
                with open(root_dir + '/{}.pkl'.format(cata), 'rb') as f:
                    dist_i = pickle.load(f)
                ax.scatter(points[i], dist_i, s=1, label=cata)
                self.flier_counter(points[i])
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
                def func(x, eta1, eta2, eta3, eta4):  # fitting curve from DISTS, suitable for SuperPoint and CES
                    return (eta1 - eta2) / (1 + np.exp(-(x - eta3) / np.abs(eta4))) + eta2

                if (p0 == np.array([1, 1, 1, 1])).all():
                    popt, pcov = curve_fit(func, flatten_points, flatten_dist)
                else:
                    popt = p0
                print('estimated params:{}'.format(popt))

            points = np.linspace(0, 1, 500)
            plt.xlim((0, 1))
            if mode == 'MSE':
                plt.xlim((0, 0.04))
            if mode == 'ORB+hamming' or mode == 'denseORB+hamming':
                points = np.linspace(0, 200, 1000)
                plt.xlim((10, 200))
            # if mode == 'denseSuperPoint+l2dist' or mode == 'denseUTR+l2dist' or mode == 'denseHardNet+l2dist':
            if mode.find('l2dist') != -1:
                points = np.linspace(0, 2, 500)
                plt.xlim((0.2, 1.5))
            if mode == 'SIFT+l2dist' or mode == 'denseSIFT+l2dist':
                plt.xlim((0, 0.15))
            if mode == 'DISTS':
                points = np.linspace(0, 0.3, 500)
                plt.xlim((0, 0.3))

            if fit:  # draw fit curve
                # print CC status
                self.compute_CC(func, popt, flatten_points, flatten_dist)
                # plot fitting curve
                plt.plot(points, func(points, *popt), 'r-', label='Fitted Curve')

            plt.ylim((-1, 21))
            plt.xlabel(mode, font1)
            plt.ylabel('Translation Distance, px', font1)
            plt.legend()
            os.makedirs(root_dir + '/dist_figs/', exist_ok=True)
            plt.savefig(root_dir + '/dist_figs/' + mode + '_distplot.tif')
            plt.show()

    def compute_similarity(self, desc_method, img_r, img_m, mode='SSIM'):
        sim = 0
        if mode == 'SSIM':
            sim = ssim(img_r, img_m)
        if mode == 'CPC' or mode == 'NCC':  # same as local NCC?
            sim = np.corrcoef(img_r.ravel(), img_m.ravel())[0, 1]
        if mode == 'MSE':
            img_r = (img_r/255).astype(np.float32)
            img_m = (img_m / 255).astype(np.float32)
            sim = mean_squared_error(img_r, img_m)
        if mode == 'CES':
            sim = compute_CES_per_patch_pair(self.desnet, img_r, img_m, step=64, kp_step=8, mode='Mean')
        # if mode == 'CES_v2':
        #     sim = compute_CES_per_patch_pair(self.desnet, img_r, img_m, step=64, kp_step=8, mode='0516')
        if mode == 'denseUTR+l2dist' or mode == 'denseSIFT+l2dist' or mode == 'denseORB+hamming':
            sim = dense_Desc_with_dist(mode, self.desnet, desc_method, img_r, img_m, step=64, kp_step=8)
        if mode == 'NCCNet+NCC':
            sim = nccnet_ncc(self.nccnet, img_r, img_m)
        if mode == 'denseSuperPoint+l2dist':
            sim = SP_l2(self.superpoint, img_r, img_m)
        if mode == 'denseHardNet+l2dist':
            sim = HD_l2(self.hardnet, img_r, img_m)
        if mode == 'denseHardNet+mcPCC':
            sim = HD_l2(self.hardnet, img_r, img_m, sim='mcPCC')
        if mode == 'denseAANet+l2dist':
            sim = AA_l2(self.aanet, img_r, img_m)
        if mode == 'DISTS':
            sim = DISTS_pt(self.dists, img_r, img_m)
        if mode == 'NMI':
            sim = normalized_mutual_information(img_r, img_m)
        return sim

    # TODO: compute PLCC, SRCC in this function
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
                def func(x, eta1, eta2, eta3, eta4):  # fitting curve from DISTS, suitable for SuperPoint and CES
                    return (eta1 - eta2) / (1 + np.exp(-(x - eta3) / np.abs(eta4))) + eta2

                if (p0 == np.array([1, 1, 1, 1])).all():
                    popt, pcov = curve_fit(func, flatten_dist, flatten_points)
                else:
                    popt = p0
                print('estimated params:{}'.format(popt))

            points = np.linspace(0, 20, 500)
            plt.ylim((0, 1))
            if mode == 'MSE':
                plt.ylim((0, 0.04))
            if mode == 'ORB+hamming' or mode == 'denseORB+hamming':
                plt.ylim((10, 200))
            if mode.find('l2dist') != -1:
                plt.ylim((0.2, 1.5))
            if mode == 'SIFT+l2dist' or mode == 'denseSIFT+l2dist':
                plt.ylim((0, 0.15))
            if mode == 'DISTS':
                plt.ylim((0, 0.3))

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
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # avoid traffic jam with CS or YY
    T4C = Test4Category()
    # root_dir = '../data/Dataset_Lucchi_v2'
    # seq_length = 615
    # basic_dir = '../data/EXP2_FlyEM/'
    # basic_dir = '../data/EXP2_FlyEM_BS/'
    basic_dir = '../data/EXP2_FlyEM_Neurite_xy16nm/'
    seq_length = 640
    # root_dir = '../data/EXP2_Lucchi/Dataset_z10nm'
    # root_dir = '../data/EXP2_Lucchi/Exp2_dataset_z30nm'
    # seq_length = 480

    # mode = 'CES'
    # compute and save points
    # for mode in ['CES',
    #              'MSE', 'SSIM', 'CPC',
    #              'denseSIFT+l2dist', 'denseORB+hamming',
    #              'NCCNet+NCC',
    #              'denseHardNet+l2dist', 'denseSuperPoint+l2dist', 'denseAANet+l2dist', 'DISTS']:
    #     T4C.test_4_cata_and_save_points(root_dir, seq_length=seq_length, mode=mode)

    # mode = 'denseHardNet+l2dist'
    # mode = 'denseAANet+l2dist'
    for z in [8, 16, 32, 64]:
        print('\n Section Thickness {}nm '.format(z))
        # root_dir = basic_dir + 'Exp2_dataset_z{}nm'.format(z)
        root_dir = basic_dir + 'Dataset_z{}nm'.format(z)
        for mode in ['CES', 'denseHardNet+l2dist', 'denseAANet+l2dist',
                     'MSE', 'SSIM', 'CPC',
                     'denseSIFT+l2dist', 'denseORB+hamming',
                     'denseSuperPoint+l2dist', 'NCCNet+NCC', 'DISTS']:
            print('Current Mode: {}'.format(mode))
            T4C.test_4_cata_and_save_points(root_dir, seq_length=seq_length, mode=mode, ver='0516')
            # draw figures
            with open(root_dir + '/points_' + mode + '.pkl', 'rb') as f:
                points = pickle.load(f)
            # T4C.plot_figures(root_dir, points, mode, fig='dist_plot', fit=False, discard_CA=False,
            #                  p0=np.array([1, 1, 1, 1]))
            T4C.plot_figures_inverse(root_dir, points, mode, fig='dist_plot', fit=True, discard_CA=False,
                                     p0=np.array([1, 1, 1, 1]))
