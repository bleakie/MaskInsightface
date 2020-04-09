import os
import cv2
import numpy as np
import argparse
from SSRNET_model import SSR_net, SSR_net_general
import csv
import math
import time
from keras import backend as K
# cudnn error
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def model_init(args):
    K.set_learning_phase(0)  # make sure its testing mode
    # load model and weights
    stage_num = [3, 3, 3]
    lambda_local = 1
    lambda_d = 1
    model_age = SSR_net(args.image_size, stage_num, lambda_local, lambda_d)()
    model_age.load_weights(args.age_model)

    model_gender = SSR_net_general(args.image_size, stage_num, lambda_local, lambda_d)()
    model_gender.load_weights(args.gender_model)
    return model_gender, model_age

def parse_args():
    parser = argparse.ArgumentParser(description='face model test')
    # general
    parser.add_argument('--image_size', default=112, type=int)
    parser.add_argument('--image', default='images', help='')
    parser.add_argument('--label', default='label.csv', help='')
    parser.add_argument('--result', default='result_age_gender.csv', help='')
    parser.add_argument('--age_model',
                        default='../pre-trained/wiki_age_bbox_crop/ssrnet_3_3_3_112_0.75_1.0.h5',
                        help='path to load model.')
    parser.add_argument('--gender_model',
                        default='../pre-trained/wiki_gender_112_class3/ssrnet_3_3_3_112_1.0_1.0.h5',
                        help='path to load model.')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    return parser.parse_args()


def get_gender_age(args, model_age, model_gender):
    count = 0.;
    age_mae = 0.; age_rmse = 0.;gender_num = 0
    output_data = []
    times = 0.
    labelList = csv.reader(open(args.label, "rt", encoding="utf-8-sig"))
    for row in labelList:
        img_id = row[0]
        true_age = int(row[2])
        true_gender = int(row[1])
        img = cv2.imread(os.path.join(args.image, img_id))
        if img is None:
            continue
        count += 1
        if count % 5000 == 0:
            print(count)
        start_time = time.time()
        face = cv2.resize(img, (args.image_size, args.image_size))
        face = face[np.newaxis, :]
        # age
        predicted_ages = float(model_age.predict(face))
        age_mae += abs(predicted_ages - true_age)
        age_rmse += math.pow(predicted_ages - true_age, 2)
        # gender
        predicted_genders = float(model_gender.predict(face))  # predicted_genders < 0.5 female
        times += (time.time() - start_time)
        if (predicted_genders < 0 and true_gender < 1) or (predicted_genders >= 0 and true_gender > 0):
            gender_num += 1
        output_data.append({'id':img_id, 'true_gender': true_gender, 'true_age': true_age,
                            'estimate_gender': predicted_genders, 'estimate_age':predicted_ages})
    age_mae = age_mae / count
    age_rmse = math.sqrt(age_rmse / count)
    gender_acc = gender_num / count
    print('Mean time:', times/count)

    with open(args.result, 'w') as f:
        headers = ['id', 'true_gender', 'true_age', 'estimate_gender', 'estimate_age']
        f_scv = csv.DictWriter(f, headers)
        f_scv.writeheader()
        f_scv.writerows(np.array(output_data))
    return gender_acc, age_mae, age_rmse

if __name__ == '__main__':
    args = parse_args()
    model_gender, model_age = model_init(args)
    gender_acc, age_mae, age_rmse = get_gender_age(args, model_age, model_gender)
    print('gender_acc:', gender_acc, 'age_mae:', age_mae, 'age_rmse:', age_rmse)

    # gender: 0.9853605560382276
    # age_mae: 3.2959095643355885
    # age_rmse: 4.256980842189304