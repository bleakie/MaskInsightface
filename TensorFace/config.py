from easydict import EasyDict as edict

config = edict()
config.INPUT = './imgs'


config.FD = edict()
config.FD.fd_conf = 0.9
config.FD.image_size = [640, 640]
config.FD.network = 'retinaface_r50_v1' # 'retinaface_mnet025_v2'

config.FR = edict()
config.FR.fr_conf = 0.3
config.FR.image_size = [112, 112]
config.FR.network = 'glint360k_r100FC_1.0'