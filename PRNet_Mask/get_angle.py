import numpy as np
import os
import cv2
import math

from PRNet_Mask.api import PRN
from PRNet_Mask.utils.estimate_pose import estimate_pose

def load_mask_model(DEVICES):
    # ---- init PRN
    os.environ['CUDA_VISIBLE_DEVICES'] = str(DEVICES) # GPU number, -1 for CPU
    prn = PRN()
    return prn

def get_angle(image, prn, max_bbox):
    if image.shape[2] > 3:
        image = image[:, :, :3]

    # the core: regress position map
    pos = prn.process(image, max_bbox)
    if pos is None:
        print('depths_img is error')
        return None
    else:
        # 3D vertices with 40k
        vertices = prn.get_vertices(pos)
        _, pose = estimate_pose(vertices, prn.canonical_vertices_40) # canonical_vertices_40: pos angle with 40k points, add by sai
        # 3D vertices with 68
        # landmarks =  np.float32(prn.get_landmarks(pos))
        # landmarks = gen_landmark(image, max_bbox)
        # canonical_vertices_fan  transform array with 68 points, which is generate by fannet
        # canonical_vertices_3d    std 3d model whit 68 points
        # _, pose = estimate_pose(landmarks, prn.canonical_vertices_40)
        angle = np.array(pose) * 180 / math.pi
        # for k in landmarks:
        #     image = cv2.circle(image, (int(k[0]), int(k[1])), 1, (255, 0, 255), 1)

        return image, angle
