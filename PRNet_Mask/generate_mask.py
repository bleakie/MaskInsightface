import numpy as np
import os
import cv2
import numpy
import time
import math
import warnings
warnings.filterwarnings("ignore")

from PRNet_Mask.api import PRN
from skimage import img_as_ubyte
from PRNet_Mask.utils.estimate_pose import estimate_pose
from PRNet_Mask.utils.render_app import get_depth_image, faceCrop

def load_mask_model(DEVICES):
    # ---- init PRN
    os.environ['CUDA_VISIBLE_DEVICES'] = str(DEVICES) # GPU number, -1 for CPU
    prn = PRN()
    print('load prnet model success')
    return prn

def add_median_colr(img_ori, img_mask, mask, min_bbox):
    img_crop = faceCrop(img_ori, min_bbox, scale_ratio=0.5)
    (B, G, R) = cv2.split(img_crop)
    B_median = np.median(B)
    G_median = np.median(G)
    R_median = np.median(R)
    # mean_pixel = cv2.mean(img[int(rect[1]):int(rect[3]), int(rect[0]):int(rect[2])])  # get img mean pixel
    rows, cols, _ = img_ori.shape
    for row in range(rows):
        for col in range(cols):
            if mask[row, col] < 1:
                img_mask[row, col][0] = B_median
                img_mask[row, col][1] = G_median
                img_mask[row, col][2] = R_median

    return img_mask


def align_face(input, preds, canonical_vertices, target_size=(318,361)):

    #get (x,y) coordinates
    canonical_vertices = canonical_vertices[:,:2]
    preds = preds[:,:2]
    front_vertices = canonical_vertices #front_vertices[:,:2]

    #get (x,y,1) coordinates
    pts0, pts1, pts2 = canonical_vertices, preds, front_vertices
    pts0_homo = np.hstack((pts0, np.ones([pts0.shape[0],1]))) #n x 4
    pts1_homo = np.hstack((pts1, np.ones([pts1.shape[0],1]))) #n x 4
    pts2_homo = np.hstack((pts2, np.ones([pts2.shape[0],1]))) #n x 4

    #get 3D transform parameters
    AM_68 = np.linalg.lstsq(pts1_homo, pts2_homo)[0].T # Affine matrix. 3 x 3
    pts1_homo_5 = np.float32([np.mean(pts1_homo[[17,36]], axis=0), np.mean(pts1_homo[[26,45]], axis=0), np.mean(pts1_homo[[30]], axis=0), np.mean(pts1_homo[[48,60]], axis=0), np.mean(pts1_homo[[54,64]], axis=0)])
    pts2_homo_5 = np.float32([np.mean(pts2_homo[[17,36]], axis=0), np.mean(pts2_homo[[26,45]], axis=0), np.mean(pts2_homo[[30]], axis=0), np.mean(pts2_homo[[48,60]], axis=0), np.mean(pts2_homo[[54,64]], axis=0)])
    AM_5 = np.linalg.lstsq(pts1_homo_5, pts2_homo_5)[0].T # Affine matrix. 3 x 3
    pts1_homo_eye = np.float32([np.mean(pts1_homo[[36]], axis=0), np.mean(pts1_homo[[39]], axis=0), np.mean(pts1_homo[[8]], axis=0), np.mean(pts1_homo[[42]], axis=0), np.mean(pts1_homo[[45]], axis=0)])
    pts2_homo_eye = np.float32([np.mean(pts2_homo[[36]], axis=0), np.mean(pts2_homo[[39]], axis=0), np.mean(pts2_homo[[8]], axis=0), np.mean(pts2_homo[[42]], axis=0), np.mean(pts2_homo[[45]], axis=0)])
    AM_eye = np.linalg.lstsq(pts1_homo_eye, pts2_homo_eye)[0].T # Affine matrix. 3 x 3
    #AM = np.median([AM_68,AM_5,AM_eye],axis=0)
    AM = (1.0*AM_68 + 2.0*AM_5 + 3.0*AM_eye) / (1.0+2.0+3.0)

    #border landmark indices
    x0_index_src = np.where(pts0[:,0]==np.amin(pts0[:,0]))[0][0]
    x1_index_src = np.where(pts0[:,0]==np.amax(pts0[:,0]))[0][0]
    y0_index_src = np.where(pts0[:,1]==np.amin(pts0[:,1]))[0][0]
    y1_index_src = np.where(pts0[:,1]==np.amax(pts0[:,1]))[0][0]

    # (x,y) limits
    x0 = pts0_homo[x0_index_src][0]
    x1 = pts0_homo[x1_index_src][0]
    y0 = pts0_homo[y0_index_src][1]
    y1 = pts0_homo[y1_index_src][1]

    # get affine transformed image
    input_crop = input #[y0:y1, x0:x1, :]
    dst = cv2.warpPerspective(input_crop, AM, target_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    #crop transformed image
    #x0 = max(0, int(x0))
    #x1 = min(dst.shape[1],int(x1))+1
    #y0 = max(0, int(y0))
    #y1 = min(dst.shape[0],int(y1))+1
    #dst_crop = dst[y0:y1, x0:x1,:]
    return dst

def generate_mask(image, prn, max_bbox, bool_mask = True):
    if image.shape[2] > 3:
        image = image[:, :, :3]

    # the core: regress position map
    pos = prn.process(image, max_bbox)  # use dlib to detect face
    if pos is None:
        print('depths_img is error')
        return None
    else:
        # 3D vertices with 68
        landmarks = np.float32(prn.get_landmarks(pos))
        if bool_mask:
            # 3D vertices with 40k
            vertices = prn.get_vertices(pos)
            h, w = image.shape[:2]
            # use depth generate mask
            depth_image = get_depth_image(vertices, prn.triangles, h, w, True)# use crender_colors plot img, so fast but not as good as render_texture
            mask = img_as_ubyte(depth_image)

            if (mask.max() <= 1 and mask.min() >= 0):
                mask = img_as_ubyte(mask)
                img_mask = image * mask[:, :, np.newaxis]
                # 空白处添加median人脸色
                img_mask = add_median_colr(image, img_mask, mask, max_bbox)
                # 直接使用3d的人脸对齐方法
                aligned = align_face(img_mask, landmarks[:, :2], prn.canonical_vertices_fan, target_size=(318, 361))
                return aligned
            else:
                print('depths_img is error')
                return None
        else:
            # 直接使用3d的人脸对齐方法
            aligned = align_face(image, landmarks, prn.canonical_vertices_fan, target_size=(318, 361))
            return aligned
