import numpy as np
import cv2
import os
import argparse
import csv


def get_args():
    parser = argparse.ArgumentParser(description="This script cleans-up noisy labels "
                                                 "and creates database for training.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=str, default="/home/sai/YANG/datasets/face_datasets/megaage_asian/megaage_asian/train_crop/",
                        help="dataset; wiki or imdb")
    parser.add_argument("--output", type=str, default='/home/sai/YANG/datasets/face_datasets/megaage_asian/megaage_asian/megaage_asian.npz',
                        help="path to output database mat file")
    parser.add_argument('--label', default='/home/sai/YANG/datasets/face_datasets/megaage_asian/megaage_asian/train.csv', help='')
    parser.add_argument("--img_size", type=int, default=112,
                        help="output image size")
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    out_genders = []
    out_ages = []
    out_imgs = []
    labelList = csv.reader(open(args.label, "rt", encoding="utf-8-sig"))
    for row in labelList:
        true_age = int(row[1])
        true_gender = int(0)
        img_id = row[0]
        img = cv2.imread(os.path.join(args.input, img_id))
        if img is None:
            continue
        out_genders.append(true_gender)
        out_ages.append(true_age)
        out_imgs.append(cv2.resize(img, (args.img_size, args.img_size)))

    np.savez(args.output, image=np.array(out_imgs), gender=np.array(out_genders), age=np.array(out_ages),
             img_size=args.img_size)


if __name__ == '__main__':
    main()
