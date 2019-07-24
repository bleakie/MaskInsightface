# -*- coding: utf-8 -*
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import argparse
import numpy as np
import random
from time import sleep

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))
from src.common import face_image
import cv2
import time

# add ssh face detection
from SSH.ssh_detector import SSHDetector

def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret

def get_max_face(bounding_boxes):
    det = bounding_boxes[:, 0:4]
    bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
    bindex = np.argmax(bounding_box_size)  # some extra weight on the centering
    return bindex

def main(args):
    # facenet.store_revision_info(src_path, output_dir, ' '.join(sys.argv))
    dataset = face_image.get_dataset('lfw', args.input_dir)
    print('dataset size', 'lfw', len(dataset))

    print('Creating networks and loading parameters')
    # load ssh detecte model
    detector = SSHDetector(args.gpu)
    # add 3D mask
    from PRNet_Mask.generate_mask import generate_mask, load_mask_model
    # load 3D mask generate model
    mask_model = load_mask_model(args.gpu)  # PRNET

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_filename = os.path.join(args.output_dir, 'train.lst')
    print('start ....')
    with open(output_filename, "w") as text_file:
        nrof_images_total = 0
        nrof = np.zeros((5,), dtype=np.int32)
        for fimage in dataset:
            if nrof_images_total % 100 == 0:
                print("Processing %d, (%s)" % (nrof_images_total, nrof))
            nrof_images_total += 1

            image_path = fimage.image_path
            if not os.path.exists(image_path):
                print('image not found (%s)' % image_path)
                continue

            try:
                img = cv2.imread(image_path)
                # print(image_path)
            except (IOError, ValueError, IndexError) as e:
                errorMessage = '{}: {}'.format(image_path, e)
                print(errorMessage)
            else:
                if img is None:
                    continue
                if img.ndim < 2:
                    print('Unable to align "%s", img dim error' % image_path)
                    continue
                if img.ndim == 2:
                    img = to_rgb(img)

                # ssh face detection
                ret = detector.detect(img, scales_index = 2)  #
                if ret is None:
                    continue

                if ret.shape[0] < 1:
                    continue
                bindex = get_max_face(ret)
                bbox = ret[bindex, :4]

                # 获取3D人脸mask
                img_mask = generate_mask(img, mask_model, bbox, args.bool_mask)
                if img_mask is None:
                    continue
                if img_mask.shape[0] != 112 and img_mask.shape[1] != 112:
                    img_mask = cv2.resize(img_mask, (112, 112), cv2.INTER_AREA)
                # cv2.imshow('warped', img_mask)
                # cv2.waitKey(1)

                _paths = fimage.image_path.split('/')
                a, b = _paths[-2], _paths[-1]
                target_dir = os.path.join(args.output_dir, a)
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                target_file = os.path.join(target_dir, b)
                cv2.imwrite(target_file, img_mask)
                oline = '%d\t%s\t%d\n' % (1, target_file, int(fimage.classname))
                text_file.write(oline)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, help='gpu id', default=0)

    parser.add_argument('--input-dir', type=str, help='Directory with unaligned images.',
                        default='/data1t/GateID/Gate_all/01')
    parser.add_argument('--output-dir', type=str, help='Directory with aligned face thumbnails.',
                        default='/data1t/GateID/gate_aligned_retina_crop')
    parser.add_argument('--bool-mask', type=int, help='Use mask.', default=0)

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
