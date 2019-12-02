import os
import cv2
import numpy as np
from skimage import transform as trans

class ARC_FACE:
    def __init__(self, image_size=112):
        arcface_src = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]], dtype=np.float32)
        self.arcface_src = np.expand_dims(arcface_src, axis=0)
        self.image_size = image_size

    # lmk is prediction; src is template
    def estimate_norm(self, lmk):
        assert lmk.shape == (5, 2)
        tform = trans.SimilarityTransform()
        lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
        min_M = []
        min_index = []
        min_error = float('inf')
        src = self.arcface_src

        for i in np.arange(src.shape[0]):
            tform.estimate(lmk, src[i])
            M = tform.params[0:2, :]
            results = np.dot(M, lmk_tran.T)
            results = results.T
            error = np.sum(np.sqrt(np.sum((results - src[i]) ** 2, axis=1)))
            #         print(error)
            if error < min_error:
                min_error = error
                min_M = M
                min_index = i
        return min_M, min_index

    def norm_crop(self, img, landmark):
        M, pose_index = self.estimate_norm(landmark)
        warped = cv2.warpAffine(img, M, (self.image_size, self.image_size), borderValue=0.0)
        return warped


class DLIB_FACE:
    def __init__(self, image_size=112):
        self.image_size = image_size
        TEMPLATE = np.float32([
            (0.0792396913815, 0.339223741112), (0.0829219487236, 0.456955367943),
            (0.0967927109165, 0.575648016728), (0.122141515615, 0.691921601066),
            (0.168687863544, 0.800341263616), (0.239789390707, 0.895732504778),
            (0.325662452515, 0.977068762493), (0.422318282013, 1.04329000149),
            (0.531777802068, 1.06080371126), (0.641296298053, 1.03981924107),
            (0.738105872266, 0.972268833998), (0.824444363295, 0.889624082279),
            (0.894792677532, 0.792494155836), (0.939395486253, 0.681546643421),
            (0.96111933829, 0.562238253072), (0.970579841181, 0.441758925744),
            (0.971193274221, 0.322118743967), (0.163846223133, 0.249151738053),
            (0.21780354657, 0.204255863861), (0.291299351124, 0.192367318323),
            (0.367460241458, 0.203582210627), (0.4392945113, 0.233135599851),
            (0.586445962425, 0.228141644834), (0.660152671635, 0.195923841854),
            (0.737466449096, 0.182360984545), (0.813236546239, 0.192828009114),
            (0.8707571886, 0.235293377042), (0.51534533827, 0.31863546193),
            (0.516221448289, 0.396200446263), (0.517118861835, 0.473797687758),
            (0.51816430343, 0.553157797772), (0.433701156035, 0.604054457668),
            (0.475501237769, 0.62076344024), (0.520712933176, 0.634268222208),
            (0.565874114041, 0.618796581487), (0.607054002672, 0.60157671656),
            (0.252418718401, 0.331052263829), (0.298663015648, 0.302646354002),
            (0.355749724218, 0.303020650651), (0.403718978315, 0.33867711083),
            (0.352507175597, 0.349987615384), (0.296791759886, 0.350478978225),
            (0.631326076346, 0.334136672344), (0.679073381078, 0.29645404267),
            (0.73597236153, 0.294721285802), (0.782865376271, 0.321305281656),
            (0.740312274764, 0.341849376713), (0.68499850091, 0.343734332172),
            (0.353167761422, 0.746189164237), (0.414587777921, 0.719053835073),
            (0.477677654595, 0.706835892494), (0.522732900812, 0.717092275768),
            (0.569832064287, 0.705414478982), (0.635195811927, 0.71565572516),
            (0.69951672331, 0.739419187253), (0.639447159575, 0.805236879972),
            (0.576410514055, 0.835436670169), (0.525398405766, 0.841706377792),
            (0.47641545769, 0.837505914975), (0.41379548902, 0.810045601727),
            (0.380084785646, 0.749979603086), (0.477955996282, 0.74513234612),
            (0.523389793327, 0.748924302636), (0.571057789237, 0.74332894691),
            (0.672409137852, 0.744177032192), (0.572539621444, 0.776609286626),
            (0.5240106503, 0.783370783245), (0.477561227414, 0.778476346951)])
        # this shape is reference from dlib implement.
        self.mean_shape_x = [0.000213256, 0.0752622, 0.18113, 0.29077, 0.393397, 0.586856, 0.689483, 0.799124,
                             0.904991, 0.98004, 0.490127, 0.490127, 0.490127, 0.490127, 0.36688, 0.426036,
                             0.490127, 0.554217, 0.613373, 0.121737, 0.187122, 0.265825, 0.334606, 0.260918,
                             0.182743, 0.645647, 0.714428, 0.793132, 0.858516, 0.79751, 0.719335, 0.254149,
                             0.340985, 0.428858, 0.490127, 0.551395, 0.639268, 0.726104, 0.642159, 0.556721,
                             0.490127, 0.423532, 0.338094, 0.290379, 0.428096, 0.490127, 0.552157, 0.689874,
                             0.553364, 0.490127, 0.42689]  # 17-67
        self.mean_shape_y = [0.106454, 0.038915, 0.0187482, 0.0344891, 0.0773906, 0.0773906, 0.0344891,
                             0.0187482, 0.038915, 0.106454, 0.203352, 0.307009, 0.409805, 0.515625, 0.587326,
                             0.609345, 0.628106, 0.609345, 0.587326, 0.216423, 0.178758, 0.179852, 0.231733,
                             0.245099, 0.244077, 0.231733, 0.179852, 0.178758, 0.216423, 0.244077, 0.245099,
                             0.780233, 0.745405, 0.727388, 0.742578, 0.727388, 0.745405, 0.780233, 0.864805,
                             0.902192, 0.909281, 0.902192, 0.864805, 0.784792, 0.778746, 0.785343, 0.778746,
                             0.784792, 0.824182, 0.831803, 0.824182]  # 17-67
        self.padding = 0.2  # change yourself
        # add padding
        new_template = []
        for item in TEMPLATE:
            new_item = (
                (item[0] + self.padding) / (2 * self.padding + 1),
                (item[1] + self.padding) / (2 * self.padding + 1))
            new_template.append(new_item)

        self.dlib_src = np.float32(new_template)

    def align_dlib_cpp(self, rgbImg, landmarks=None):
        '''
        @brief: 与dlib C++版本实现的裁剪对齐方法一致。
        @attention
        '''
        assert rgbImg is not None

        npLandmarks = np.array(landmarks)[:, :2]
        shape_x = [npLandmarks[i][0] for i in range(68)]
        shape_y = [npLandmarks[i][1] for i in range(68)]
        from_points = []
        to_points = []
        for i in range(17, 68):
            # 忽略掉低于嘴唇的部分
            if i >= 55 and i <= 59:
                continue
            # 忽略眉毛部分
            if i >= 17 and i <= 26:
                continue
            # 上下左右都padding
            new_ref_x = (self.padding + self.mean_shape_x[i - 17]) / (2 * self.padding + 1)
            new_ref_y = (self.padding + self.mean_shape_y[i - 17]) / (2 * self.padding + 1)

            from_points.append((shape_x[i], shape_y[i]))
            to_points.append((self.image_size * new_ref_x, self.image_size * new_ref_y))

        source = np.array(from_points).astype(np.int)
        target = np.array(to_points, ).astype(np.int)
        source = np.reshape(source, (1, 36, 2))
        target = np.reshape(target, (1, 36, 2))
        H = cv2.estimateRigidTransform(source, target, False)
        if H is None:
            return None
        else:
            aligned_face = cv2.warpAffine(rgbImg, H, (self.image_size, self.image_size))
            return aligned_face

class GH_FACE:
    def __init__(self, image_size=112):
        self.front_size = (318, 361)
        self.image_size = image_size
        self.canonical_vertices = np.load(os.path.join(os.path.dirname(__file__), "canonical_vertices_male.npy"))

    def align_face_gh(self, input, preds):
        # get (x,y) coordinates
        canonical_vertices = self.canonical_vertices[:, :2]
        preds = preds[:, :2]
        front_vertices = canonical_vertices  # front_vertices[:,:2]

        # get (x,y,1) coordinates
        pts0, pts1, pts2 = canonical_vertices, preds, front_vertices
        pts0_homo = np.hstack((pts0, np.ones([pts0.shape[0], 1])))  # n x 4
        pts1_homo = np.hstack((pts1, np.ones([pts1.shape[0], 1])))  # n x 4
        pts2_homo = np.hstack((pts2, np.ones([pts2.shape[0], 1])))  # n x 4
        # get 3D transform parameters
        AM_68 = np.linalg.lstsq(pts1_homo, pts2_homo)[0].T  # Affine matrix. 3 x 3
        pts1_homo_5 = np.float32([np.mean(pts1_homo[[17, 36]], axis=0), np.mean(pts1_homo[[26, 45]], axis=0),
                                  np.mean(pts1_homo[[30]], axis=0), np.mean(pts1_homo[[48, 60]], axis=0),
                                  np.mean(pts1_homo[[54, 64]], axis=0)])
        pts2_homo_5 = np.float32([np.mean(pts2_homo[[17, 36]], axis=0), np.mean(pts2_homo[[26, 45]], axis=0),
                                  np.mean(pts2_homo[[30]], axis=0), np.mean(pts2_homo[[48, 60]], axis=0),
                                  np.mean(pts2_homo[[54, 64]], axis=0)])
        AM_5 = np.linalg.lstsq(pts1_homo_5, pts2_homo_5)[0].T  # Affine matrix. 3 x 3
        pts1_homo_eye = np.float32(
            [np.mean(pts1_homo[[36]], axis=0), np.mean(pts1_homo[[39]], axis=0), np.mean(pts1_homo[[8]], axis=0),
             np.mean(pts1_homo[[42]], axis=0), np.mean(pts1_homo[[45]], axis=0)])
        pts2_homo_eye = np.float32(
            [np.mean(pts2_homo[[36]], axis=0), np.mean(pts2_homo[[39]], axis=0), np.mean(pts2_homo[[8]], axis=0),
             np.mean(pts2_homo[[42]], axis=0), np.mean(pts2_homo[[45]], axis=0)])
        AM_eye = np.linalg.lstsq(pts1_homo_eye, pts2_homo_eye)[0].T  # Affine matrix. 3 x 3
        # AM = np.median([AM_68,AM_5,AM_eye],axis=0)
        AM = (1.0 * AM_68 + 2.0 * AM_5 + 3.0 * AM_eye) / (1.0 + 2.0 + 3.0)

        # get affine transformed image
        dst = cv2.warpPerspective(input, AM, self.front_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        # # border landmark indices
        # x0_index_src = np.where(pts0[:, 0] == np.amin(pts0[:, 0]))[0][0]
        # x1_index_src = np.where(pts0[:, 0] == np.amax(pts0[:, 0]))[0][0]
        # y0_index_src = np.where(pts0[:, 1] == np.amin(pts0[:, 1]))[0][0]
        # y1_index_src = np.where(pts0[:, 1] == np.amax(pts0[:, 1]))[0][0]
        #
        # # (x,y) limits
        # x0 = pts0_homo[x0_index_src][0]
        # x1 = pts0_homo[x1_index_src][0]
        # y0 = pts0_homo[y0_index_src][1]
        # y1 = pts0_homo[y1_index_src][1]
        # # crop transformed image
        # x0 = max(0, int(x0))
        # x1 = min(dst.shape[1], int(x1)) + 1
        # y0 = max(0, int(y0))
        # y1 = min(dst.shape[0], int(y1)) + 1
        # dst_crop = dst[y0:y1, x0:x1, :]

        aligned_face = cv2.resize(dst, (self.image_size, self.image_size), cv2.INTER_AREA)
        return aligned_face