# -*- coding: utf-8 -*
import cv2
import sys
import math
import numpy as np
import csv
import os


def read_img(path_img):
    img = cv2.imread(path_img)
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img


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


def get_all_path_sort(file_dir):
    files = os.listdir(file_dir)  # 采用listdir来读取所有文件
    files.sort()  # 排序
    s = []  # 创建一个空列表
    for file_ in files:
        list = []  # 循环读取每个文件名
        if file_.endswith('jpg'):
            f_name = str(file_)
            list.append(({"name": f_name}, {"path": os.path.join(file_dir, f_name)}))
            s.append(list)
    return s


# get top index sim
def get_top(fea, feature, top=2, COLOR='bgr'):
    index_name = []
    index_sim = []
    color_index = 'fea_' + COLOR
    fea_index = []
    for j in feature:
        sim = np.dot(fea, j[color_index].T)
        index_sim.append(sim)
        fea_index.append(j[color_index])
        index_name.append(j['id'])
    index_sim = np.array(index_sim)
    order = index_sim.argsort()[::-1]
    index_name = np.array(index_name)
    NAME = index_name[order][:top]
    SIM = index_sim[order][:top]
    fea_index = np.array(fea_index)
    fea_index = fea_index[order][:top]
    return NAME[0], SIM[0], fea_index


# get top same id
import collections
def bool_reranking(NAME, SIM, top_num=10):
    sample_name = NAME[:top_num]
    sample_sim = np.array(SIM[:top_num])
    dic = collections.Counter(sample_name)

    top_nums = list(dic.values())
    top_ids = list(dic.keys())
    ids_index, sim_index = [], []
    for i in range(len(top_ids)):
        score = 0.0
        for j in range(len(sample_name)):
            if top_ids[i] == sample_name[j]:
                score += sample_sim[j]
        ids_index.append(top_ids[i])
        sim = score/(top_nums[i]+1) # 按照出现的次数，越多权重越大
        sim_index.append(sim)
    sim_index = np.array(sim_index)
    order = sim_index.argsort()[::-1]
    sim_index = sim_index[order]
    ids_index = np.array(ids_index)
    ids_index = ids_index[order]
    data = (ids_index, sim_index)
    return data

# get score updata
def ScoreAug(score, aug_threshold):
    k1 = (4 - 1) / (0.98 - aug_threshold)
    b1 = 4 - 0.98 * k1
    k2 = (-1 - -4) / (aug_threshold - 0)
    b2 = -1 - aug_threshold * k2

    if score > aug_threshold:
        return 1.0 / (1.0 + np.e ** (-(k1 * score + b1)))
    else:
        return 1.0 / (1.0 + np.e ** (-(k2 * score + b2)))
