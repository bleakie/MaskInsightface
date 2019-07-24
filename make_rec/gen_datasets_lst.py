# -*- coding: utf-8 -*
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import argparse
import numpy as np
import random
from time import sleep

sys.path.append(os.path.join(os.path.dirname(__file__), 'common'))
from common import face_image
import cv2
import time


def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


def main(args):
    # facenet.store_revision_info(src_path, output_dir, ' '.join(sys.argv))
    dataset = face_image.get_dataset('lfw', args.input_dir)
    print('dataset size', 'lfw', len(dataset))

    output_filename = os.path.join(args.output_dir, 'train.lst')

    with open(output_filename, "w") as text_file:
        nrof_images_total = 0
        nrof = np.zeros((5,), dtype=np.int32)
        for fimage in dataset:
            if nrof_images_total % 50000 == 0:
                print("Processing %d, (%s)" % (nrof_images_total, nrof))
            nrof_images_total += 1

            _paths = fimage.image_path.split('/')
            a, b = _paths[-2], _paths[-1]
            target_dir = os.path.join(args.input_dir, a)
            # if not os.path.exists(target_dir):
            #     os.makedirs(target_dir)
            target_file = os.path.join(target_dir, b)
            oline = '%d\t%s\t%d\n' % (1, target_file, int(fimage.classname))
            text_file.write(oline)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--input-dir', type=str, help='Directory with unaligned images.',
                        default='/data1t/mask/glint-mask')
    parser.add_argument('--output-dir', type=str, help='Directory with aligned face thumbnails.',
                        default='/data1t/mask/mask-output')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
