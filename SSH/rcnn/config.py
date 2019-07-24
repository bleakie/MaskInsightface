import numpy as np
from easydict import EasyDict as edict

config = edict()

config.TEST = edict()

# RCNN nms
config.TEST.NMS = 0.3

config.TEST.SCORE_THRESH = 0.85

# scale changed as smallhard face
config.TEST.SCALES = [100, 1000]#[50,100,300,600,]
config.TEST.PYRAMID_SCALES = [1.0]
config.TEST.CONSTANT = 30
# default settings
default = edict()
