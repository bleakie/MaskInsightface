# -*- coding: utf-8 -*-
from keras.models import load_model
import numpy as np
import os
import cv2
from FaceQNet import load_Qnet_model, face_quality

# Loading the pretrained model
model = load_Qnet_model()

IMG_PATH = '/home/sai/YANG/image/video/nanning/haha'
dir = os.listdir(IMG_PATH)
count = len(dir)
print('count:', count)
for i in dir:
    count -= 1
    if count%1000==0:
        print('count:', count)
    dir_path = os.path.join(IMG_PATH, i)
    imgs_dir = os.listdir(dir_path)
    for j in imgs_dir:
        img_path = os.path.join(dir_path, j)
        img = cv2.imread(img_path)
        score = face_quality(model, img)
        # img = [cv2.resize(cv2.imread(img_path, cv2.IMREAD_COLOR), (224, 224))]
        # test_data = np.array(img, copy=False, dtype=np.float32)
        # score = model.predict(test_data, batch_size=1, verbose=1)
        path1 = str(score[0][0]) + '@'
        rename = path1 + j
        os.rename(img_path, os.path.join(dir_path, rename))
