import time
import os
import numpy as np
import cv2
import sys
import numpy as np
from numpy.linalg import norm
sys.path.append(os.path.dirname(__file__))
from modules.model_zoo.getter import get_model

class FR:
    def __init__(self, model_name = 'glint360k_r100FC_1.0', model_path = './models'):
        print('FR init...')
        self.model = get_model(model_name, root_dir=model_path, force_fp16=False)
        self.model.prepare()

    def normalize(self, img):
        embedding = self.model.get_embedding(img)[0].tolist()
        embedding_norm = norm(embedding)
        normed_embedding = embedding / embedding_norm
        return normed_embedding


if __name__ == '__main__':
    fr = FR()
    img1 = cv2.imread('/media/sai/机械2/FACE/Test_datasets/img_ours/retina-5/02/wangwenhuan.jpg', cv2.IMREAD_COLOR)
    emb1 = fr.normalize(img1)

    import os
    imgList = os.listdir('/media/sai/机械2/FACE/Test_datasets/img_ours/retina-5/02')
    for im in imgList:
        img2 = cv2.imread(os.path.join('/media/sai/机械2/FACE/Test_datasets/img_ours/retina-5/02', im), cv2.IMREAD_COLOR)
        emb2 = fr.normalize(img2)
        sim = np.dot(emb1, emb2)
        print(im, sim)


