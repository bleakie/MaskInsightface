import os
import cv2
import time
import argparse
import sys
import numpy as np
sys.path.append(os.path.dirname(__file__))
from modules.imagedata import ImageData
from common.face_preprocess import preprocess
from config import config
from fd_retina import FD
from fr_arcface import FR


def face_align(src_img, fd, args):
    src_image = ImageData(src_img, fd.input_shape)
    src_image.resize_image(mode='pad')

    dets, lmks = fd.detector.detect(src_image.transformed_image, threshold=args.fd_conf)
    if dets.shape[0] > 0:
        bindex = fd.get_max_face(dets)
        landmark = lmks[bindex]
        bbox = dets[bindex, :4].astype(np.int)
        image_size = '%d,%d' % (args.fr_input_size[1], args.fr_input_size[0])
        warped = preprocess(src_image.transformed_image, bbox, landmark, image_size = image_size)
        return warped
    else:
        return None


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, help='gpu id', default=0)
    parser.add_argument('--input', type=str, help='Directory with unaligned images.',
                        default=config.INPUT)
    # FD
    parser.add_argument('--fd_input_size', default=config.FD.image_size)
    parser.add_argument('--fd_model', default=config.FD.network, help='path to load model.')
    parser.add_argument('--fd_conf', default=config.FD.fd_conf, help='path to load model.')
    # FR
    parser.add_argument('--fr_input_size', default=config.FR.image_size)
    parser.add_argument('--fr_model', default=config.FR.network, help='path to load model.')
    parser.add_argument('--fr_conf', default=config.FR.fr_conf, help='path to load model.')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    fd = FD()
    fr = FR()
    idList = os.listdir(args.input)
    for id in idList:
        imgList = os.listdir(os.path.join(args.input, id))
        start = time.time()
        for im in imgList:
            src_img = cv2.imread(os.path.join(args.input, id, 'src.jpg'), cv2.IMREAD_COLOR)
            card_img = cv2.imread(os.path.join(args.input, id, 'card.jpg'), cv2.IMREAD_COLOR)
            if src_img is None or card_img is None:
                continue
            src_warped = face_align(src_img, fd, args)
            if src_warped is None:
                continue
            card_warped = face_align(card_img, fd, args)
            if card_warped is None:
                continue
            src_emb = fr.normalize(src_warped)
            card_emb = fr.normalize(card_warped)
            sim = np.dot(src_emb, card_emb)
            print(id, sim)

