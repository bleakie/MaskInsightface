#!/usr/bin/env python
# coding: utf-8

# # Face Quality Assessment for Face Verification in Video 
# https://pdfs.semanticscholar.org/2c0a/caec54ab2585ff807e18b6b9550c44651eab.pdf?_ga=2.118968650.2116578973.1552199994-98267093.1547624592
import cv2
import numpy as np


# get illumination
def illumination(img, bbox):
    bbox = bbox.astype(np.int)
    gray = cv2.cvtColor(img[bbox[1]:bbox[3], bbox[0]:bbox[2], :], cv2.COLOR_BGR2GRAY)
    # length of R available  range  of  gray  intensities  excluding  5%  of  the darkest  and  brightest  pixel
    sorted_gray = np.sort(gray.ravel())
    l = len(sorted_gray)
    cut_off_idx = l * 5 // 100
    r = sorted_gray[l - cut_off_idx] - sorted_gray[cut_off_idx]
    return np.round(r / 255, 2)


def get_contour(pts):
    return np.array([[pts[i], pts[5 + i]] for i in [0, 1, 4, 3]], np.int32).reshape((-1, 1, 2))


def get_mask(image, contour):
    mask = np.zeros(image.shape[0:2], dtype="uint8")
    cv2.drawContours(mask, [contour], -1, 255, -1)
    return mask


# get sharpness
def sharpness(img, lmk):
    x_index, y_index = [], []
    for i in lmk:
        x_index.append(i[0])
        y_index.append(i[1])
    landmark = np.append(x_index, y_index)
    contour = get_contour(landmark)
    mask = get_mask(img, contour)  # 1-channel mask
    mask = np.stack((mask,) * 3, axis=-1)  # 3-channel mask
    mask[mask == 255] = 1  # convert 0 and 255 to 0 and 1
    laplacian = cv2.Laplacian(img, cv2.CV_64F)

    edges = laplacian[mask.astype(bool)]
    return np.round(edges.var() / 255, 2)


# get size
def get_size(bbox, lower_threshold = 60):
    x = min(bbox[2] - bbox[0], bbox[3] - bbox[1])

    if (x > lower_threshold):
        return False
    else:
        return True


def check_large_pose(landmark, bbox):
    assert landmark.shape == (5, 2)
    assert len(bbox) == 4

    def get_theta(base, x, y):
        vx = x - base
        vy = y - base
        vx[1] *= -1
        vy[1] *= -1
        tx = np.arctan2(vx[1], vx[0])
        ty = np.arctan2(vy[1], vy[0])
        d = ty - tx
        d = np.degrees(d)
        if d < -180.0:
            d += 360.
        elif d > 180.0:
            d -= 360.0
        return d

    landmark = landmark.astype(np.float32)

    theta1 = get_theta(landmark[0], landmark[3], landmark[2])
    theta2 = get_theta(landmark[1], landmark[2], landmark[4])
    # print(va, vb, theta2)
    theta3 = get_theta(landmark[0], landmark[2], landmark[1])
    theta4 = get_theta(landmark[1], landmark[0], landmark[2])
    theta5 = get_theta(landmark[3], landmark[4], landmark[2])
    theta6 = get_theta(landmark[4], landmark[2], landmark[3])
    theta7 = get_theta(landmark[3], landmark[2], landmark[0])
    theta8 = get_theta(landmark[4], landmark[1], landmark[2])
    # print(theta1, theta2, theta3, theta4, theta5, theta6, theta7, theta8)
    left_score = 0.0
    right_score = 0.0
    up_score = 0.0
    down_score = 0.0
    if theta1 <= 0.0:
        left_score = 10.0
    elif theta2 <= 0.0:
        right_score = 10.0
    else:
        left_score = theta2 / theta1
        right_score = theta1 / theta2
    if theta3 <= 10.0 or theta4 <= 10.0:
        up_score = 10.0
    else:
        up_score = max(theta1 / theta3, theta2 / theta4)
    if theta5 <= 10.0 or theta6 <= 10.0:
        down_score = 10.0
    else:
        down_score = max(theta7 / theta5, theta8 / theta6)
    print(left_score, right_score , up_score, down_score)
    if left_score < 8 and right_score < 8 and up_score < 3 and down_score < 3:
        return False
    else:
        return True

def over_border(img, landmark):
    h, w = img.shape[:2]
    xmin, xmax = min(landmark[:, 0]), max(landmark[:, 0])
    ymin, ymax = min(landmark[:, 1]), max(landmark[:, 1])
    if min(xmin, ymin) < 0:
        return True
    elif xmax > w or ymax > h:
        return True
    else:
        return False

def faceCrop(img, maxbbox, scale_ratio=1.0):
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
    return [xmin, ymin, xmax, ymax]

def get_face_quality(img, face_bbox, landmark):
    small_size = get_size(face_bbox)  # size > 0
    # score_sharpness = sharpness(img, landmark)  # 0.3
    # score_illumination = illumination(img, face_bbox)  # 0.5
    out_border = over_border(img, landmark)
    large_pose = check_large_pose(landmark, face_bbox)

    if small_size or out_border:
        return False
    elif large_pose:
        return False
    # elif min(score_sharpness, score_illumination) < 0.1:
    #     return False
    else:
        return True

def get_person_quality(dress_bbox, face_bbox):
    face_hight = face_bbox[3]-face_bbox[1]
    dress_hight = dress_bbox[3]-dress_bbox[1]
    if dress_hight / face_hight < 0.8: # dress
        return False
    else:
        return True