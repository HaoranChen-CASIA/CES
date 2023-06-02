import numpy as np
import cv2


def encoding_patch(row, col, img, des, step=64, kp_step=8):
    kp = []
    for i_local in range(row * step, (row + 1) * step, kp_step):
        for j_local in range(col * step, (col + 1) * step, kp_step):
            kp.append(cv2.KeyPoint(32 + j_local, 32 + i_local, 64))  # opencv point (x, y), W to H

    # step 2: compute utr desc encodings for current patch pair
    l_utr = np.array(des.compute(img, kp)).T
    return l_utr


def l_utr_statistic(l_ref, l_mov, mode):
    sta_lst = []
    dim = l_ref.shape[0]
    for z in range(dim):
        l_r_dim = l_ref[z, :]
        l_m_dim = l_mov[z, :]
        utr_zncc = abs(np.corrcoef(l_r_dim.ravel(), l_m_dim.ravel())[0, 1])
        sta_lst.append(utr_zncc)
    encoding_zncc = np.array(sta_lst)

    res = 0
    if mode == 'Mean':
        if np.any(np.isnan(encoding_zncc)):
            res = 0
        else:
            res = encoding_zncc.mean()
    if mode == '0516':
        Q3 = np.percentile(encoding_zncc, 75)
        Q1 = np.percentile(encoding_zncc, 25)
        encoding_zncc[encoding_zncc > Q3] = 0
        encoding_zncc[encoding_zncc < Q1] = 0
        res = encoding_zncc.sum()/(128 - np.sum(encoding_zncc == 0))

    return res


def compute_CES_per_patch_pair(des, img_r, img_m, step, kp_step, mode='Mean'):
    # step 2: encode patch using des_net
    l_ref = encoding_patch(0, 0, img_r, des, step, kp_step)
    l_mov = encoding_patch(0, 0, img_m, des, step, kp_step)

    # step 3: save current heatmap value
    encoding_similarity = l_utr_statistic(l_ref, l_mov, mode=mode)

    return encoding_similarity

