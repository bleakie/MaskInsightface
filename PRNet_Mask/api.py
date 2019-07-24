import numpy as np
import os
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp
from time import time

from PRNet_Mask.predictor import PosPrediction

import cv2
class PRN:
    ''' Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network
    Args:
        is_dlib(bool, optional): If true, dlib is used for detecting faces.
        prefix(str, optional): If run at another folder, the absolute path is needed to load the data.
    '''
    def __init__(self, is_dlib = False):

        # resolution of input and output image size.
        self.resolution_inp = 256
        self.resolution_op = 256

        prefix = os.path.dirname(__file__)
        #---- load detectors
        if is_dlib:
            import dlib
            detector_path = os.path.join(prefix, 'Data/net-data/mmod_human_face_detector.dat')
            self.face_detector = dlib.cnn_face_detection_model_v1(
                    detector_path)

        #---- load PRN
        self.pos_predictor = PosPrediction(self.resolution_inp, self.resolution_op)
        prn_path = os.path.join(prefix, 'Data/net-data/256_256_resfcn256_weight')
        if not os.path.isfile(prn_path + '.data-00000-of-00001'):
            print("please download PRN trained model first.")
            exit()
        self.pos_predictor.restore(prn_path)

        # uv file
        self.uv_kpt_ind = np.loadtxt(prefix + '/Data/uv-data/uv_kpt_ind.txt').astype(np.int32) # 2 x 68 get kpt
        self.face_ind = np.loadtxt(prefix + '/Data/uv-data/face_ind.txt').astype(np.int32) # get valid vertices in the pos map
        self.triangles = np.loadtxt(prefix + '/Data/uv-data/triangles.txt').astype(np.int32) # ntri x 3
        self.canonical_vertices_fan = (np.load(prefix + '/Data/uv-data/canonical_vertices_68_fan.npy'))
        self.canonical_vertices_40 = np.load(prefix + '/Data/uv-data/canonical_vertices_40k.npy') # pos angle with 40k point, add by sai

        self.uv_coords = self.generate_uv_coords()

    def generate_uv_coords(self):
        resolution = self.resolution_op
        uv_coords = np.meshgrid(range(resolution),range(resolution))
        uv_coords = np.transpose(np.array(uv_coords), [1,2,0])
        uv_coords = np.reshape(uv_coords, [resolution**2, -1]);
        uv_coords = uv_coords[self.face_ind, :]
        uv_coords = np.hstack((uv_coords[:,:2], np.zeros([uv_coords.shape[0], 1])))
        return uv_coords

    def net_forward(self, image):
        ''' The core of out method: regress the position map of a given image.
        Args:
            image: (256,256,3) array. value range: 0~1
        Returns:
            pos: the 3D position map. (256, 256, 3) array.
        '''
        return self.pos_predictor.predict(image)

    def faceCrop(self, img, maxbbox, scale_ratio=2):
        '''
        crop face from image, the scale_ratio used to control margin size around face.
        using a margin, when aligning faces you will not lose information of face
        '''
        xmin, ymin, xmax, ymax = maxbbox
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
        face = img[ymin:ymax, xmin:xmax, :]
        return face

    def process(self, image, bbox):
        ''' process image with crop operation.
        Args:
            input: (h,w,3) array or str(image path). image value range:1~255.
            image_info(optional): the bounding box information of faces. if None, will use dlib to detect face.

        Returns:
            pos: the 3D position map. (256, 256, 3).
        '''
        if image.ndim < 3:
            image = np.tile(image[:, :, np.newaxis], [1, 1, 3])

        left = bbox[0];
        right = bbox[2]
        top = bbox[1]
        bottom = bbox[3]
        old_size = (right - left + bottom - top) / 2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 + old_size * 0.14])
        size = int(old_size * 1.318)

        # crop image
        src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                            [center[0] + size / 2, center[1] - size / 2]])
        DST_PTS = np.array([[0, 0], [0, self.resolution_inp - 1], [self.resolution_inp - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)

        image = image / 255.
        cropped_image = warp(image, tform.inverse, output_shape=(self.resolution_inp, self.resolution_inp))

        # run our net
        # st = time()
        cropped_pos = self.net_forward(cropped_image)
        # print 'net time:', time() - st
        crop_pos = cropped_pos.copy()
        # restore
        cropped_vertices = np.reshape(cropped_pos, [-1, 3]).T

        z = cropped_vertices[2, :].copy() / tform.params[0, 0]
        cropped_vertices[2, :] = 1
        vertices = np.dot(np.linalg.inv(tform.params), cropped_vertices)
        vertices = np.vstack((vertices[:2, :], z))
        pos = np.reshape(vertices.T, [self.resolution_op, self.resolution_op, 3])

        return pos

    def get_landmarks(self, pos):
        '''
        Args:
            pos: the 3D position map. shape = (256, 256, 3).
        Returns:
            kpt: 68 3D landmarks. shape = (68, 3).
        '''
        kpt = pos[self.uv_kpt_ind[1,:], self.uv_kpt_ind[0,:], :]
        return kpt


    def get_vertices(self, pos):
        '''
        Args:
            pos: the 3D position map. shape = (256, 256, 3).
        Returns:
            vertices: the vertices(point cloud). shape = (num of points, 3). n is about 40K here.
        '''
        all_vertices = np.reshape(pos, [self.resolution_op**2, -1])
        vertices = all_vertices[self.face_ind, :]

        return vertices

    def get_colors_from_texture(self, texture):
        '''
        Args:
            texture: the texture map. shape = (256, 256, 3).
        Returns:
            colors: the corresponding colors of vertices. shape = (num of points, 3). n is 45128 here.
        '''
        all_colors = np.reshape(texture, [self.resolution_op**2, -1])
        colors = all_colors[self.face_ind, :]

        return colors


    def get_colors(self, image, vertices):
        '''
        Args:
            pos: the 3D position map. shape = (256, 256, 3).
        Returns:
            colors: the corresponding colors of vertices. shape = (num of points, 3). n is 45128 here.
        '''
        [h, w, _] = image.shape
        vertices[:,0] = np.minimum(np.maximum(vertices[:,0], 0), w - 1)  # x
        vertices[:,1] = np.minimum(np.maximum(vertices[:,1], 0), h - 1)  # y
        ind = np.round(vertices).astype(np.int32)
        colors = image[ind[:,1], ind[:,0], :] # n x 3

        return colors