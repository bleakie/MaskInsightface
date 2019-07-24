# -*- coding: utf-8 -*
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import numpy as np
import mxnet as mx
import cv2
import sklearn

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# add ssh face detection
from SSH.ssh_detector import SSHDetector
# add 3D mask
from PRNet_Mask.generate_mask import generate_mask, load_mask_model

def do_flip(data):
    for idx in range(data.shape[0]):
        data[idx, :, :] = np.fliplr(data[idx, :, :])


def get_max_face(bounding_boxes):
    det = bounding_boxes[:, 0:4]
    bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
    bindex = np.argmax(bounding_box_size)  # some extra weight on the centering
    return bindex


def get_model(ctx, image_size, model_str, layer):
    _vec = model_str.split(',')
    assert len(_vec) == 2
    prefix = _vec[0]
    epoch = int(_vec[1])
    print('loading', prefix, epoch)
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    all_layers = sym.get_internals()
    sym = all_layers[layer + '_output']
    model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
    model.set_params(arg_params, aux_params)
    return model


class FaceModel:
    def __init__(self, args):
        self.args = args
        ctx = mx.gpu(args.gpu)
        _vec = args.image_size.split(',')
        assert len(_vec) == 2
        image_size = (int(_vec[0]), int(_vec[1]))
        self.model = None
        self.ga_model = None
        if len(args.model) > 0:
            self.model = get_model(ctx, image_size, args.model, 'fc1')
        # if len(args.ga_model)>0:
        #   self.ga_model = get_model(ctx, image_size, args.ga_model, 'fc1')

        self.det_minsize = 50
        self.image_size = image_size
        # load 68 landmark model
        self.landmark_net = load_mask_model(args.gpu)
        # 使用ssh人脸检测
        self.detector = SSHDetector(args.gpu, False)

    def get_input(self, face_img):
        ret = self.detector.detect(face_img, scales_index=2)
        if ret is None or ret.shape[0] < 1:
            return None
        bindex = get_max_face(ret)
        bbox = ret[bindex, :4]
        # 获取3D人脸mask
        warped = generate_mask(face_img, self.landmark_net, bbox, True)
        if warped is None:
            return None
        nimg = cv2.resize(warped, (112, 112), cv2.INTER_AREA)
        # face_score = face_quality(self.Qnet, nimg)

        RGB_nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)  # train img is all bgr
        aligned_rgb = np.transpose(RGB_nimg, (2, 0, 1))
        return aligned_rgb

    def get_feature(self, aligned):
        input_blob = np.expand_dims(aligned, axis=0)
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        self.model.forward(db, is_train=False)
        embedding = self.model.get_outputs()[0].asnumpy()
        embedding = sklearn.preprocessing.normalize(embedding).flatten()
        return embedding

    def get_ga(self, aligned):
        input_blob = np.expand_dims(aligned, axis=0)
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        self.ga_model.forward(db, is_train=False)
        ret = self.ga_model.get_outputs()[0].asnumpy()
        g = ret[:, 0:2].flatten()
        gender = np.argmax(g)
        a = ret[:, 2:202].reshape((100, 2))
        a = np.argmax(a, axis=1)
        age = int(sum(a))

        return gender, age
