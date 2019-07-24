# -*- coding: utf-8 -*-
from keras.models import load_model
import numpy as np
import os
import cv2

def load_Qnet_model():
	# os.environ['CUDA_VISIBLE_DEVICES']='0'
	base_path = os.path.dirname(__file__)
	#Loading the pretrained model
	model = load_model(base_path+'/FaceQnet.h5')
	return model

def face_quality(model, img):
	img = cv2.resize(img, (224, 224))
	test_data = []
	test_data.append(img)
	test_data = np.array(test_data, copy=False, dtype=np.float32)
	#Extract quality scores for the samples
	score = model.predict(test_data, batch_size=1, verbose=1)
	return score
