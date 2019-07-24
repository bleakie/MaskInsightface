# -*- coding: utf-8 -*
import face_model_video
import argparse
import cv2
import sys
import math
import os
import numpy as np

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='../models/glink-best/model,1', help='path to load model.')
# parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
parser.add_argument('--face_detection', default=1, help='0 is mtcnn, 1 is ssh')
parser.add_argument('--ssh_model', default='../SSH/model/model-zhaohang-v1.0/e2e', type=str, help='0 is mtcnn, 1 is ssh')
parser.add_argument('--face_landmark', default='../face_landmark_xiaoyi/model/ssh_FAN-4_003.pth',
                    help='load 68 landmark')
args = parser.parse_args()

model = face_model_video.FaceModel(args)


def get_all_path(IMAGE_DIR):
    image_path = []
    g = os.walk(IMAGE_DIR)
    for path, d, filelist in g:
        for filename in filelist:
            list = []
            if filename.endswith('jpg'):
                list.append(({"name": filename}, {"path": os.path.join(path, filename)}))
                image_path.append(list)
    return image_path
#
IMAGE_DIR = '/home/sai/YANG/image/Face_Recognition/landmark-data/other-src'
SAVE_DIR = '/home/sai/YANG/image/Face_Recognition/landmark-data/'

count = 0
file_names = get_all_path(IMAGE_DIR)
feature=[]
for index in range(len(file_names)):
    name = file_names[index][0][0]["name"]
    file_path = file_names[index][0][1]["path"]
    img = cv2.imread(file_path)
    img_draw = img.copy()
    # if count%5!=0:
    #     continue
    # print(file_path)
    aligned = model.get_input(img)
    if aligned is None:
        continue
    if len(aligned) != 1:
        continue
    for i in aligned:
        bbox = i['bbox']
        # cv2.rectangle(img_draw, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 1)
        f1 = model.get_feature(i['aligned'])
        f1 = list(f1)
        for k in i['landmark']:
            f1.append(k[0])
            f1.append(k[1])
            cv2.circle(img_draw, (int(k[0]), int(k[1])), 1, (255, 0, 255), 1)

        feature.append({'id': 'false', 'fea': np.array(f1)})
    count += 1
    # cv2.imwrite(os.path.join(SAVE_DIR+'src-new', str(count)+'_new.jpg'), img)
    # cv2.imwrite(os.path.join(SAVE_DIR + 'other-draw', name), img_draw)

np.save(SAVE_DIR+'/other_feature.npy', feature)


