#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: save_heatmap.py
# Created Date: Friday January 15th 2021
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Wednesday, 19th January 2022 1:22:47 am
# Modified By: Chen Xuanhong
# Copyright (c) 2021 Shanghai Jiao Tong University
#############################################################

import os
import shutil
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import numpy as np

def SaveHeatmap(heatmaps, path, row=-1, dpi=72):
    """
    The input tensor must be B X 1 X H X W
    """
    batch_size = heatmaps.shape[0]
    temp_path  = ".temp/"
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)
    final_img = None
    if row < 1:
        col = batch_size
        row = 1
    else:
        col = batch_size // row
        if row * col <batch_size:
            col +=1
    
    row_i = 0
    col_i = 0
    
    for i in range(batch_size):
        img_path = os.path.join(temp_path,'temp_batch_{}.png'.format(i))
        sns.heatmap(heatmaps[i,0,:,:],vmin=0,vmax=heatmaps[i,0,:,:].max(),cbar=False)
        plt.savefig(img_path, dpi=dpi, bbox_inches = 'tight', pad_inches = 0)
        img = cv2.imread(img_path)
        if i == 0:
            H,W,C = img.shape
            final_img = np.zeros((H*row,W*col,C))
        final_img[H*row_i:H*(row_i+1),W*col_i:W*(col_i+1),:] = img
        col_i += 1
        if col_i >= col:
            col_i = 0
            row_i += 1
    cv2.imwrite(path,final_img)

if __name__ == "__main__":
    random_map = np.random.randn(16,1,10,10)
    SaveHeatmap(random_map,"./wocao.png",1)
