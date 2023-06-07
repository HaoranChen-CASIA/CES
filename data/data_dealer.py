import time

import cv2
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import random

import numba as nb
import pickle
from multiprocessing import Pool
import tifffile as tiff


class DataDealer:
    def __init__(self, args):
        self.src_path = args.src_path

    @staticmethod
    def down_sample_as_wish(src_path, psy_res, img_format, start_layer, end_layer, crop=False, crop_size=[], z_step=1, zfill=4):
        down_scale = psy_res[2] // psy_res[0]
        # down_path = src_path + 'down_scale{}/zstep{}/'.format(down_scale, z_step)
        down_path = src_path + 'down_scale{}'.format(down_scale, z_step)
        if crop:
            down_path += '_crop/'
        else:
            down_path += '/'
        os.makedirs(down_path, exist_ok=True)
        j = 0
        for i in range(start_layer, end_layer + 1, z_step):
            i_org = cv2.imread(src_path + str(i).zfill(zfill) + img_format,
                               0)  # 0 for uint8 raw image, CV_16UC1 for neuron_ids
            i_down = cv2.resize(i_org, (i_org.shape[1] // down_scale, i_org.shape[0] // down_scale),
                                interpolation=cv2.INTER_CUBIC)
            if crop:
                h_c = crop_size[0]//2
                w_c = crop_size[1]//2
                h = i_down.shape[0]//2  # height center
                w = i_down.shape[1]//2  # width center
                i_down = i_down[h-h_c:h+h_c, w-w_c:w+w_c]
            cv2.imwrite(down_path + str(j).zfill(4) + '.png', i_down)
            # cv2.imwrite(down_path + str(i).zfill(4) + '.png', i_down)
            j += 1

    @staticmethod
    def tps_transform(img, n_ids, grid_img, std=10, portion=250):
        """
        :param img: original image before warp
        :return: image warped by random tps transform
        """
        tps = cv2.createThinPlateSplineShapeTransformer()
        tps.setRegularizationParameter(0.1)
        shape = img.shape
        # create sshape uniformly distributed in the image
        num_y = shape[0] // portion
        num_x = shape[1] // portion
        x = np.linspace(0, shape[1], num_x)
        y = np.linspace(0, shape[0], num_y)
        xx, yy = np.meshgrid(x, y)
        sshape = np.zeros((num_x * num_y, 2))
        sshape[:, 0] = xx.flatten()
        sshape[:, 1] = yy.flatten()
        sshape = sshape.astype(np.float32)
        # create tshape based on sshape, using normal distribution with mean and std
        random_vector = np.random.normal(0, std, sshape.shape).astype(np.float32)
        # # save random_vector
        # with open(self.vector_path + '/random_vector' + str(idx).zfill(3) + '.pkl', 'wb') as f:
        #     pickle.dump(random_vector, f)
        tshape = random_vector + sshape
        sshape = sshape.reshape(1, -1, 2)
        tshape = tshape.reshape(1, -1, 2)
        matches = list()
        for i in range(sshape.shape[1]):
            matches.append(cv2.DMatch(i, i, 0))
        tps.estimateTransformation(tshape, sshape, matches)

        out_img = tps.warpImage(img, flags=cv2.INTER_LINEAR)  # XinT: do not use nearest for raw image
        out_grid_img = tps.warpImage(grid_img, flags=cv2.INTER_LINEAR)  # XinT: do not use nearest for raw image
        out_ids = tps.warpImage(n_ids, flags=cv2.INTER_LINEAR)

        return out_img, out_ids, out_grid_img  #, random_vector

    @staticmethod
    def conventional_transform(img, img_ids, grid_img, mode, angle=361, std=10, mean=0):
        rows, cols = img.shape
        flag = 0

        # define a random deformation metric based on mode
        if mode == 'affine' or 'translation':
            p1 = np.float32([[0, 0], [cols//2, 0], [0, rows//2]])
            # create a random vector using normal distribution with zero-mean and std
            if mode == 'translation':
                trans_vec = np.random.normal(mean, std, size=(1, 2)).astype(np.float32)
                dist = np.linalg.norm(trans_vec)
                if dist <= 6:
                    print('Slight Translation! ')
                    flag = 2
                if dist > 6 and dist <= 10:
                    print('Poor Translation! ')
                    flag = 1
                random_vector = np.array([trans_vec, trans_vec, trans_vec]).squeeze(1)
            else:
                random_vector = np.random.normal(mean, std, size=p1.shape).astype(np.float32)
            p2 = p1 + random_vector
            deform_mat = cv2.getAffineTransform(p1, p2)
        if mode == 'rotation':
            center = (cols // 2, rows // 2)
            if angle == 361:
                angle = np.random.randint(low=-15, high=15)
            if abs(angle) <= 5:
                print('Slight Rotation!')
                flag = 2
            if abs(angle) > 5 and abs(angle) <= 10:
                print('Poor Rotation!')
                flag = 1
            deform_mat = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)

        # warp original image
        warped_img = cv2.warpAffine(img, deform_mat, dsize=(cols, rows))
        warped_grid = cv2.warpAffine(grid_img, deform_mat, dsize=(cols, rows))
        warped_ids = cv2.warpAffine(img_ids, deform_mat, dsize=(cols, rows))


        return warped_img, warped_ids, warped_grid, flag

    @staticmethod
    @nb.jit()
    def CURT(img_f, img_m):
        img_curt = np.zeros(img_m.shape, dtype=np.uint8)
        # N_f = img_f.shape[0] * img_f.shape[1]
        # N_m = img_m.shape[0] * img_f.shape[1]
        sorted_f = np.sort(img_f.ravel())
        sorted_m = np.sort(img_m.ravel())

        for i in range(img_f.shape[0]):
            for j in range(img_f.shape[1]):
                n_f = np.where(sorted_f == img_f[i, j])[0][0]
                img_curt[i, j] = sorted_m[n_f]

        return img_curt

    def construct_dataset_chunk_wise(self, range_s, range_e, z_dist=6, step=16, category='CA',
                                     sample_dist=26, datatype='raw'):
        """
        :param range_s: start section
        :param range_e: end section
        :param category: 1 in 4 categories, CA, SA, PA and WA
        :return: image chunks of corresponding category
        """
        ver = 2
        cate_path = self.src_path + 'Dataset_v{}/'.format(ver) + category + '/'
        os.makedirs(cate_path, exist_ok=True)
        ref_path = self.src_path + 'Dataset_v{}/ref/'.format(ver)
        os.makedirs(ref_path, exist_ok=True)
        print('Creating Dataset')
        j = 0
        dist_lst = []
        for i in range(range_s, range_e, sample_dist):
            if datatype == 'raw':
                img_r = cv2.imread(self.src_path + str(i).zfill(4) + '.png', 0)
                img_m = cv2.imread(self.src_path + str(i + z_dist).zfill(4) + '.png', 0)
            else:  # read in labels
                img_r = tiff.imread(self.src_path + str(i).zfill(4) + '.tif')
                img_m = tiff.imread(self.src_path + str(i + z_dist).zfill(4) + '.tif')
            for patch_x in range(84, 256-84, step):
                for patch_y in range(84, 320-84, step):
                    patch_r = self.extract_chunk(img_r, patch_x, patch_y)
                    if category == 'CA':
                        if datatype == 'raw':
                            cv2.imwrite(ref_path + str(j).zfill(3) + '.png', patch_r)
                        else:
                            tiff.imwrite(ref_path + str(j).zfill(3) + '.tif', patch_r)
                    dist, dx, dy = self.set_translation(category)
                    patch_m = self.extract_chunk(img_m, patch_x + dx, patch_y + dy)
                    if datatype == 'raw':
                        cv2.imwrite(cate_path + str(j).zfill(3) + '.png', patch_m)
                    else:
                        tiff.imwrite(cate_path + str(j).zfill(3) + '.tif', patch_m)
                    dist_lst.append(dist)
                    j += 1
        print('Category {}, distance list:{}'.format(category, dist_lst))
        with open(self.src_path + 'Dataset_v{}/{}'.format(ver, category) + '.pkl', 'wb') as f1:
            pickle.dump(dist_lst, f1)

    @staticmethod
    def extract_chunk(img, x, y, patch_size=64):
        """
        :param img: source image
        :param x: patch center
        :param y: patch center
        :param patch_size: patch size, default 64
        :return: extracted image patch
        """
        img_p = cv2.copyMakeBorder(img, 32, 32, 32, 32, cv2.BORDER_CONSTANT, 0)
        patch = img_p[x + 32 - patch_size:x + 32 + patch_size, y + 32 - patch_size:y + 32 + patch_size]
        return patch

    @staticmethod
    def set_translation(category):
        # TODO: change to randint
        # set dx and dy
        if category == 'CA':
            dist = 0
            dx = 0
            dy = 0
        else:
            if category == 'SA':
                dist = np.random.uniform(1, 6)
            if category == 'PA':
                dist = np.random.uniform(6, 10)
            if category == 'WA':
                dist = np.random.uniform(10, 20)
            sign_dx = random.choice((-1, 1))
            sign_dy = random.choice((-1, 1))
            dx = int(sign_dx * np.random.uniform(1, dist))
            dy = int(sign_dy * pow((dist * dist - dx * dx), 0.5))
        return dist, dx, dy


def rename_and_reindex(src_path, start_old, end_old, z_fill=3):
    reindex_path = src_path + 'reindexed/'
    os.makedirs(reindex_path, exist_ok=True)
    for i in range(start_old, end_old + 1):
        img = cv2.imread(src_path + 'layer{}'.format(i) + '/1_1.bmp', 0)
        # img = cv2.imread(src_path + 'layer{}'.format(i) + '.tif', 0)
        # new_idx = i - start_old + 1
        new_idx = i - start_old
        cv2.imwrite(reindex_path + str(new_idx).zfill(z_fill) + '.png', img)


def create_grid(size, path, grid_size):
    num_x, num_y = (size[1]//grid_size, size[0]//grid_size)
    x, y = np.meshgrid(np.linspace(0, size[1], num_y + 1), np.linspace(0, size[0], num_x + 1))

    plt.figure(figsize=(size[1]/100.0, size[0]/100.0))
    plt.plot(y, x, color='black')
    plt.plot(y.transpose(), x.transpose(), color='black')
    plt.axis('off')

    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)

    plt.savefig(path + 'grid.png')


def artificial_deformation(DD, src_path, seq_length):
    # combine affine and tps transform
    raw_path = src_path + 'origin/down_xy8_z1/'
    ids_path = src_path + 'neuron_A/down_xy8_z1/'
    raw_out_path = raw_path + 'deformed/'
    ids_out_path = ids_path + 'deformed/'
    grid_out_path = raw_out_path + 'warped_grid/'
    os.makedirs(ids_out_path, exist_ok=True)
    os.makedirs(grid_out_path, exist_ok=True)

    # create grid_image
    create_grid([128, 128], src_path, grid_size=64)
    grid_img = cv2.imread(src_path + 'grid.png', 0)

    i = 1
    z_step = 1
    slight_lst = []
    poor_lst = []
    worse_lst = []
    while i < seq_length and i + z_step <= seq_length:
        print('\n section {}'.format(i+z_step))
        flag_t = 0
        flag_r = 0
        # img_ref = cv2.imread(src_path + str(i).zfill(4) + '.png', 0)
        img_mov = cv2.imread(raw_path + str(i + z_step).zfill(4) + '.png', 0)
        img_ids = cv2.imread(ids_path + str(i).zfill(4) + '.png', cv2.CV_16UC1)
        warped, warped_ids, warped_grid, flag_t = DD.conventional_transform(img_mov, img_ids, grid_img,
                                                                            mode='translation', std=5, mean=0)
        warped, warped_ids, warped_grid, flag_r = DD.conventional_transform(warped, warped_ids, warped_grid,
                                                                            mode='rotation')
        # warped, warped_ids, warped_grid = DD.tps_transform(warped, warped_ids, warped_grid, std=10, portion=64)
        # cv2.imwrite(out_path + str(i).zfill(4) + '.png', img_ref)
        cv2.imwrite(raw_out_path + str(i + z_step).zfill(4) + '.png', warped)
        cv2.imwrite(ids_out_path + str(i).zfill(4) + '.png', warped_ids)
        cv2.imwrite(grid_out_path + str(i + z_step).zfill(4) + '.png', warped_grid)
        flag = flag_t + flag_r
        if flag == 4:
            slight_lst.append(i + z_step)
        elif flag == 3:
            poor_lst.append(i + z_step)
        else:
            worse_lst.append(i + z_step)
        i += 1  # z_distance = 5*6 = 30nm
    return slight_lst, poor_lst, worse_lst


def generate_gif(path, save_name):
    import imageio
    import glob

    images = []
    for filename in glob.glob(path + '*.tif'):
        images.append(cv2.imread(filename, 0))

    imageio.mimsave(save_name + '.gif', images, fps=5)


def get_parser():
    parser = argparse.ArgumentParser(description='side-view slice generator')
    # src_path: the path of the folder containing the original image files
    parser.add_argument('-s', '--src_path', type=str,
                        default='D:/chenhr/CAS_design/TestCodes/Python_Code/DataSet/',
                        help='Source directory')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_parser()
    # args.src_path, datatype = './data/Lucchi++/Lucchi_dxy6_z1/', 'raw'
    # z_step, step, seqlength = 2, 32, 1060  # thickness = z_step*z_step, x-y step = step pixels
    # #
    # # # # args.src_path, datatype = './data/FlyEM_32nm_raw/', 'raw'
    # # # args.src_path, datatype = './data/FlyEM_32nm_labels/', 'label'
    # # # z_step, step, seqlength = 1, 32, 200
    # #
    # DD = DataDealer(args)
    #
    # for catagory in ['CA', 'SA', 'PA', 'WA']:
    #     DD.construct_dataset_chunk_wise(1, seqlength, z_step, step, catagory, sample_dist=40, datatype=datatype)

    args.src_path = './data/FlyEM_8nm_raw/'
    DD = DataDealer(args)
    DD.down_sample_as_wish(args.src_path, [8, 8, 16], '.png', 0, 799, z_step=1)




