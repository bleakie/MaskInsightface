# -*- coding: utf-8 -*-
import numpy as np
import os
import cv2
import shutil

IMG_PATH = '/data1t/glint-gate-retina-crop'
High_PATH = '/data1t/glint_datasets/glint-aligned-v4-quality'
dir = os.listdir(IMG_PATH)
count = len(dir)
print('count:', count)
for i in dir:
    count -= 1
    if count%1000==0:
        print('count:', count)
    dir_path = os.path.join(IMG_PATH, i)
    imgs_dir = os.listdir(dir_path)

    save_path = os.path.join(High_PATH, i)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for j in imgs_dir:
        img_path = os.path.join(dir_path, j)
        if not os.path.exists(img_path):
            continue
        path_index = j.index('@')
        score = float(j[:path_index])
        if score > 0.45:
            shutil.copy(img_path, save_path)
