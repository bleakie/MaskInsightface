import numpy as np
import cv2
import os
import csv
import argparse
from tqdm import tqdm
from TYY_utils import get_meta


def get_args():
    parser = argparse.ArgumentParser(description="This script cleans-up noisy labels "
                                                 "and creates database for training.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output", "-o", type=str, default='/home/sai/YANG/datasets/face_datasets/wiki/',
                        help="path to output database mat file")
    parser.add_argument("--input", type=str, default="/home/sai/YANG/datasets/face_datasets/wiki/gender_age_bbox_crop/",
                        help="dataset; wiki or imdb")
    parser.add_argument("--img_size", type=int, default=112,
                        help="output image size")
    parser.add_argument("--min_score", type=float, default=1.0,
                        help="minimum face_score")
    parser.add_argument("--label", type=str, default='/home/sai/YANG/datasets/face_datasets/wiki/label.csv',
                        help="path to output database mat file")
    args = parser.parse_args()
    return args


def main_gender_age():
    args = get_args()
    labelList = csv.reader(open(args.label, "rt", encoding="utf-8-sig"))
    out_ages = []
    out_imgs = []
    out_genders = []
    for row in labelList:
        true_age = row[2]
        true_gender = row[1]
        img_id = row[0]
        img_path = os.path.join(args.input, img_id)
        img = cv2.imread(img_path)
        if img is None:
            continue
        out_genders.append(int(true_gender))
        out_ages.append(int(true_age))
        out_imgs.append(cv2.resize(img, (args.img_size, args.img_size)))
    print('len:', len(out_imgs))
    np.savez('train_data/wiki_bbox_crop.npz',image=np.array(out_imgs), gender=np.array(out_genders), age=np.array(out_ages), img_size=args.img_size)


def main_csv():
    args = get_args()
    output_path = args.output
    db = args.input
    min_score = args.min_score

    mat_path = os.path.join(db, "{}.mat".format('wiki'))
    full_path, dob, gender, photo_taken, face_score, second_face_score, age = get_meta(mat_path, 'wiki')
    output_data = []
    for i in tqdm(range(len(face_score))):
        if face_score[i] < min_score:
            continue
        if (~np.isnan(second_face_score[i])) and second_face_score[i] > 0.0:
            continue
        if ~(0 <= age[i] <= 100):
            continue
        if np.isnan(gender[i]):
            continue
        img_id = str(full_path[i][0])

        save_path = os.path.join(output_path, img_id.split('/')[1])
        img_path = os.path.join(db, str(full_path[i][0]))
        import shutil
        shutil.copy(img_path, save_path)
        output_data.append({'gender':gender[i], 'age':age[i], 'id': img_id.split('/')[1]})
    with open(args.label, 'w') as f:
        headers = ['id', 'gender', 'age']
        f_scv = csv.DictWriter(f, headers)
        f_scv.writeheader()
        f_scv.writerows(np.array(output_data))

if __name__ == '__main__':
    main_gender_age()
