import numpy as np
import os
from glob import glob
import scipy.io as sio
from skimage.io import imread, imsave
from skimage.transform import rescale, resize
from time import time
import argparse
import ast
import sys

from PRNet_Mask.api import PRN

# from PRNet_Mask.utils.estimate_pose import estimate_pose
# from PRNet_Mask.utils.rotate_vertices import frontalize
# from PRNet_Mask.utils.render_app import get_visibility, get_uv_mask, get_depth_image
# from PRNet_Mask.utils.write import write_obj_with_colors, write_obj_with_texture


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# add ssh face detection
from SSH.ssh_detector import SSHDetector
from SSH.rcnn.config import config

# load ssh detecte model
detector = SSHDetector(gpu=0, test_mode=False)


def get_max_face(bounding_boxes):
    det = bounding_boxes[:, 0:4]
    bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
    bindex = np.argmax(bounding_box_size)  # some extra weight on the centering
    return bindex


def main(args):
    # ---- init PRN
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu # GPU number, -1 for CPU
    prn = PRN(is_dlib = args.isDlib)

    # ------------- load data
    image_folder = args.inputDir
    save_folder = args.outputDir
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    types = ('*.jpg', '*.png', '*.bmp')
    image_path_list= []
    for files in types:
        image_path_list.extend(glob(os.path.join(image_folder, files)))

    for i, image_path in enumerate(image_path_list):

        name = image_path.strip().split('/')[-1][:-4]

        if os.path.exists(os.path.join(save_folder, name + '_mesh.mat')):
            continue
        try:
            image = imread(image_path)
        except:
            continue
        if len(image.shape)<3:
            continue
        [h, w, c] = image.shape
        if c>3:
            image = image[:,:,:3]

        max_size = max(image.shape[0], image.shape[1])
        if max_size> 800:
            image = rescale(image, 800./max_size)
            image = (image*255).astype(np.uint8)
        # the core: regress position map
        if args.isDlib:
            pos = prn.process(image) # use dlib to detect face
        else:
            ret = detector.detect(image, threshold=config.TEST.SCORE_THRESH, scales=config.TEST.PYRAMID_SCALES)
            # ssh face detection
            if ret is None:
                continue
            if ret.shape[0] < 1:
                continue
            bindex = get_max_face(ret)
            bbox = ret[bindex, :4]
            pos = prn.process(image, bbox)
        
        image = image/255.
        if pos is None:
            continue

            # 3D vertices
        vertices = prn.get_vertices(pos)

        if args.isMat:
            # corresponding colors
            colors = prn.get_colors(image, vertices)
            sio.savemat(os.path.join(save_folder, name + '_mesh.mat'), {'vertices': vertices, 'colors': colors, 'triangles': prn.triangles})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network')

    parser.add_argument('-i', '--inputDir', default='TestImages/', type=str,
                        help='path to the input directory, where input images are stored.')
    parser.add_argument('-o', '--outputDir', default='TestImages/results', type=str,
                        help='path to the output directory, where results(obj,txt files) will be stored.')
    parser.add_argument('--gpu', default='0', type=str,
                        help='set gpu id, -1 for CPU')
    parser.add_argument('--isDlib', default=False, type=ast.literal_eval,
                        help='whether to use dlib for detecting face, default is True, if False, the input image should be cropped in advance')
    parser.add_argument('--isMat', default=False, type=ast.literal_eval,
                        help='whether to save vertices,color,triangles as mat for matlab showing')
    main(parser.parse_args())
