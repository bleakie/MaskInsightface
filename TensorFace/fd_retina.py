import os
import cv2
import time
import sys
import numpy as np
sys.path.append(os.path.dirname(__file__))
from modules.imagedata import ImageData
from modules.model_zoo.getter import get_model

class FD:
    def __init__(self, model_name = 'retinaface_r50_v1', model_path = './models', im_size=[640, 640], fd_conf = 0.5):
        # model_name = 'retinaface_mnet025_v2'
        print('FD init...')
        self.detector = get_model(model_name, im_size=im_size, root_dir=model_path, force_fp16=False)
        self.detector.prepare(nms=0.4)
        self.input_shape = self.detector.input_shape[2:][::-1]
        self.fd_conf = fd_conf

    def detect(self, image):
        image = ImageData(image, self.input_shape)
        image.resize_image(mode='pad')

        dets, lmks = self.detector.detect(image.transformed_image, threshold=self.fd_conf)
        DETS, LMKS = [], []
        for i in range(len(dets)):
            face = self.faceCrop(image, dets[i])
            if min(face[2], face[3]) > 0:
                DETS.append(face)
                LMKS.append(lmks[i])
        return np.array(DETS), np.array(LMKS)

    def faceCrop(self, img, maxbbox, scale_ratio=1.0):
        '''
        crop face from image, the scale_ratio used to control margin size around face.
        using a margin, when aligning faces you will not lose information of face
        '''
        xmin, ymin, xmax, ymax, score = maxbbox
        hmax, wmax, _ = img.shape
        x = (xmin + xmax) / 2
        y = (ymin + ymax) / 2
        w = (xmax - xmin) * scale_ratio
        h = (ymax - ymin) * scale_ratio
        # new xmin, ymin, xmax and ymax
        xmin = x - w / 2
        xmax = x + w / 2
        ymin = y - h / 2
        ymax = y + h / 2

        xmin = max(0, int(xmin))
        ymin = max(0, int(ymin))
        xmax = min(wmax, int(xmax))
        ymax = min(hmax, int(ymax))
        return [xmin, ymin, xmax, ymax, score]

    def get_max_face(self, bounding_boxes):
        det = bounding_boxes[:, 0:4]
        bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
        bindex = np.argmax(bounding_box_size)  # some extra weight on the centering
        return bindex



if __name__ == '__main__':
    fd = FD()
    INPUT = '/media/sai/机械1/datasets/face/anti_spoofing/手机银行/imgs/XSB'
    imgList = os.listdir(INPUT)
    start = time.time()
    for im in imgList:
        image = cv2.imread(os.path.join(INPUT, im), cv2.IMREAD_COLOR)
        image = ImageData(image, fd.input_shape)
        image.resize_image(mode='pad')

        dets, lmks = fd.detector.detect(image.transformed_image, threshold=0.5)

        for i in range(dets.shape[0]):
            landmark5 = lmks[i].astype(np.int)
            for l in range(landmark5.shape[0]):
                cv2.circle(image.transformed_image, (landmark5[l][0], landmark5[l][1]), 1, (0, 255, 255), 2)
            box = dets[i].astype(np.int)
            cv2.rectangle(image.transformed_image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        cv2.imshow('', image.transformed_image)
        cv2.waitKey()
    print((time.time()-start)/len(imgList))
