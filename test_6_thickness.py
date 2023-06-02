import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.util import view_as_windows
import cv2

from Judge.test_4_category import Test4Category


def draw_boxplot(data, mode, split=200):
    data = data.ravel()
    points = []
    for i in range(0, 6):
        d = data[i*split:(i+1)*split]
        points.append(d)
        flier_counter(d)

    fig, ax = plt.subplots()
    dict = ax.boxplot(points, showmeans=True)
    ax.set_xticklabels(['5', '10', '20', '30', '40', '50'])

    font1 = {'weight': 'bold',
             'style': 'normal',
             'size': 12,
             }

    plt.ylim((0, 1.2))
    if mode == 'MSE':
        plt.ylim((0, 0.03))
    if mode == 'denseSIFT+l2dist':
        plt.ylim((0, 0.12))
    if mode == 'denseORB+hamming':
        plt.ylim((0, 120))
    if mode == 'DISTS':
        plt.ylim((0, 0.25))
    if mode == 'NMI':
        plt.ylim((1.0, 1.2))
    plt.xlabel('Section Thickness, nm', font1)
    plt.ylabel(mode, font1)
    os.makedirs('../data/thickness_figs', exist_ok=True)
    plt.savefig('../data/thickness_figs/' + mode + '_thickplot.tif')
    plt.show()


def flier_counter(data):
    median = np.median(data)
    mean = np.mean(data)
    std = np.std(data)
    # print('median {}, mean {}'.format(median, mean))
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    Up_lim = Q3 + 1.5*IQR
    Down_lim = Q1 - 1.5*IQR
    # Up_lim = mean + 2 * std
    # Down_lim = mean - 2 * std
    cnt = np.sum(data < Down_lim) + np.sum(data > Up_lim)
    # print('filer num {}'.format(cnt))
    print('OR {}%'.format(cnt/len(data)*100))


def test_on_2_section(T4C, layer_ref, layer_mov, mode):
    if mode == 'denseSIFT+l2dist':
        desc_method = T4C.sift
    else:
        desc_method = T4C.orb
    if mode in ['MSE', 'SSIM', 'CPC', 'NMI', 'NCCNet+NCC', 'denseSuperPoint+l2dist', 'DISTS']:
        patch_size = (64, 64)
        stride = (64, 64)
    else:
        layer_ref = np.pad(layer_ref, ((32, 32), (32, 32)), 'constant', constant_values=0)
        layer_mov = np.pad(layer_mov, ((32, 32), (32, 32)), 'constant', constant_values=0)
        patch_size = (128, 128)
        stride = (64, 64)
    # divide into patches
    patches_ref = view_as_windows(layer_ref, patch_size, step=stride)
    patches_mov = view_as_windows(layer_mov, patch_size, step=stride)
    # check registration quality
    n_rows, n_cols, _, _ = patches_ref.shape
    data_i = []
    for row in range(n_rows):
        for col in range(n_cols):
            img_r = patches_ref[row, col]
            img_m = patches_mov[row, col]
            data_i.append(T4C.compute_similarity(desc_method, img_r, img_m, mode))
    data_i = np.array(data_i).ravel()
    return data_i


def test_on_fixed_thickness(T4C, root_dir, seq_length, mode, z_step):
    print('z_step = {} nm'.format(z_step * 5))
    data = []
    for i in range(1, seq_length, 100):
        layer_ref = cv2.imread(root_dir + '/' + str(i).zfill(4) + '.png', 0)
        layer_mov = cv2.imread(root_dir + '/' + str(i + z_step).zfill(4) + '.png', 0)
        data_i = test_on_2_section(T4C, layer_ref, layer_mov, mode)
        data.append(data_i)
    data = np.array(data).ravel()
    return data


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # avoid traffic jam
    T4C = Test4Category()
    root_dir = '../data/mito_dxy6_z1_c'
    seq_length = 1000

    mode = 'denseUTR+l2dist'
    data = []
    print('Current mode: {}'.format(mode))
    for z_step in [1, 2, 4, 6, 8, 10]:  # z_distance [5, 10, 20, 30, 40, 50]nm
        data.append(test_on_fixed_thickness(T4C, root_dir, seq_length, mode, z_step))
    data = np.array(data)
    draw_boxplot(data, mode, split=200)

