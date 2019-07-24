# -*- coding: utf-8 -*
import face_model_prnet_mask
import argparse
import cv2
import sys
import math
import numpy as np
import csv
import os

from deploy.FUNCTION import *

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='../model-resnet152-quality-mask/model,1', help='path to load model.')
parser.add_argument('--gpu', default=0, type=str, help='gpu id')
parser.add_argument('--flip', default=1, type=int, help='whether do lr flip aug')

args = parser.parse_args()

model = face_model_prnet_mask.FaceModel(args)

IMAGE_DIR = '../images/test_imgs/false'
txt_write = open('../images/test/result.txt', 'w')
feature = np.load('../images/test/all_img_resnet152_mask.npy')

file_names = get_all_path(IMAGE_DIR)
for index in range(len(file_names)):
    name = file_names[index][0][0]["name"]
    file_path = file_names[index][0][1]["path"]
    img = cv2.imread(file_path)

    aligned_rgb = model.get_input(img)
    if aligned_rgb is None:
        continue
    fea = model.get_feature(aligned_rgb)
    NAME, SIM, _ = get_top(fea, feature, top=1, COLOR='rgb')

    # print(name, NAME, SIM, face_score)
    txt_write.write(str(name) + ' ')
    txt_write.write(str(NAME[0]) + ' ')
    txt_write.write(str(SIM[0]) + '\n')
txt_write.close()

#
# IMAGE_DIR = '../images/test/20190430_ids/'
# feature = []
# dir0 = os.listdir(IMAGE_DIR)
# count = len(dir0)
# for i in dir0:
#     count -= 1
#     if count%2000 == 0:
#         print(count)
#
#     file_path = os.path.join(IMAGE_DIR+i)
#     img = cv2.imread(file_path)
#     if img is None:
#         continue
#     aligned_rgb = model.get_input(img)
#     if aligned_rgb is None:
#         continue
#
#     f_rgb = model.get_feature(aligned_rgb)
#     feature.append({'id':i.strip('.jpg'), 'fea_rgb':f_rgb})
# np.save('../images/test/all_img_resnet152_mask.npy', feature)