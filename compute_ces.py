import os
from CES.ces import *
from CES.desNet import Des

import argparse
import cv2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=str, default='./data/example/chunk_ref.png')
    parser.add_argument("--moving", type=str, default='./data/example/chunk_mov.png')
    args = parser.parse_args()
    
    chunk_ref = cv2.imread(args.reference, 0)
    chunk_mov = cv2.imread(args.moving, 0)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    desnet = Des(device=device)

    sim = compute_CES_per_patch_pair(self.desnet, img_r, img_m, step=64, kp_step=8, mode='Mean')
    
    print('CES value for matching-pair: {}'.format(sim))
    
    
