# -*- coding: utf-8 -*
# 筛选同文件夹下不同的人
import face_model
import argparse
import cv2
import sys
import os
import numpy as np

import shutil

def get_all_path(IMAGE_DIR):
    image_path = []
    g = os.walk(IMAGE_DIR)
    for path, d, filelist in g:
        for filename in filelist:
            list = []
            if filename.endswith('jpg') or filename.endswith('bmp'):
                list.append(({"name": filename}, {"path": os.path.join(path, filename)}))
                image_path.append(list)
    return image_path

def review_img(feature, img_index):
    # 计算与其他img的相似度，平均相似度小于0.5删除
    del_index = []
    for i in range(len(feature)):
        sim = 0.
        for j in range(len(feature)):
            if i != j:
                sim += np.dot(feature[i], feature[j].T)
        sim = sim / (len(feature) - 1)
        if sim < 0.25:
            del_index.append(i)
    for k in del_index:
        os.remove(img_index[k])
        # print(img_index[k], 'is removed')

def main(args):
    model = face_model.FaceModel(args)
    # 打开文件
    dir0 = os.listdir(args.image_path)
    count = len(dir0)
    for id in dir0:
        count -= 1
        if count%3000 == 0:
            print('last num:', count)
        name_path = os.path.join(args.image_path, id)
        name_files = get_all_path(name_path)
        if len(name_files) < 4:
            shutil.rmtree(name_path)
            # print(args.image_path + id, 'is < 1 size')
            continue
        else:
            feature = []
            img_index = []
            for index in range(len(name_files)):
                file_path = name_files[index][0][1]["path"]
                img = cv2.imread(file_path)
                if img.ndim<2:
                    os.remove(file_path)
                    continue
                # print(file_path)
                aligned = model.get_input(img)
                if aligned is None:
                    os.remove(file_path)
                    continue
                else:
                    f1 = model.get_feature(aligned)
                    feature.append(f1)
                    img_index.append(file_path)
            review_img(feature, img_index)
            name_files = os.listdir(name_path)
            if len(name_files) < 2: # 清洗后的数量小于20删除该文件夹
                shutil.rmtree(name_path)
                # print(args.image_path + id, ' is < 1 size')


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='face model test')
    # general
    parser.add_argument('--image-size', default='112,112', help='')
    parser.add_argument('--image_path', default='/home/sai/YANG/image/Face_Recognition/GateFace/raw/50000_out/', help='')
    parser.add_argument('--model', default='../models/glink-best/model,1', help='path to load model.')
    parser.add_argument('--ga-model', default='../models/gamodel-r50/model,0', help='path to load model.')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
    parser.add_argument('--flip', default=1, type=int, help='whether do lr flip aug')
    parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
    parser.add_argument('--face_detection', default=1, help='0 is mtcnn, 1 is ssh')
    parser.add_argument('--ssh_model', default='../SSH/model/model-zhaohang-v1.0/e2e', type=str, help='0 is mtcnn, 1 is ssh')
    parser.add_argument('--face_landmark', default='../face_landmark_xiaoyi/model/ssh_FAN-4_003.pth',
                        help='load 68 landmark')
    # args = parser.parse_args()
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

