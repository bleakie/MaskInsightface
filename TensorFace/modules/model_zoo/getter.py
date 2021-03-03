import os
from typing import List

from .face_detectors import *
from .face_processors import *
from ..utils.helpers import prepare_folders
from ..configs import Configs
from .exec_backends import trt_backend

# Map model names to corresponding functions
models = {
    'arcface_r100_v1': arcface_r100_v1,
    'r50-arcface-msfdrop75': r50_arcface_msfdrop75,
    'r100-arcface-msfdrop75': r100_arcface_msfdrop75,
    'glint360k_r100FC_1.0': glint360k_r100FC_1_0,
    'glint360k_r100FC_0.1': glint360k_r100FC_0_1,
    'genderage_v1': genderage_v1,
    'retinaface_r50_v1': retinaface_r50_v1,
    'retinaface_mnet025_v1': retinaface_mnet025_v1,
    'retinaface_mnet025_v2': retinaface_mnet025_v2,
    'mnet_cov2': mnet_cov2,
    'centerface': centerface,
    'dbface': dbface,
}


def prepare_backend(model_name, im_size: List[int] = None,
                    max_batch_size: int = 1,
                    force_fp16: bool = False,
                    download_model: bool = True,
                    config: Configs = None):
    """
    Check if ONNX, MXNet and TensorRT models exist and download/create them otherwise.

    :param model_name: Name of required model. Must be one of keys in `models` dict.
    :param backend_name: Name of inference backend. (onnx, trt)
    :param im_size: Desired maximum size of image in W,H form. Will be overridden if model doesn't support reshaping.
    :param max_batch_size: Maximum batch size for inference, currently supported for ArcFace model only.
    :param force_fp16: Force use of FP16 precision, even if device doesn't support it. Be careful. TensorRT specific.
    :param download_model: Download MXNet or ONNX model if it not exist.
    :param config:  Configs class instance
    :return: ONNX model serialized to string, or path to TensorRT engine
    """

    prepare_folders([config.mxnet_models_dir, config.onnx_models_dir, config.trt_engines_dir])

    in_package = config.in_official_package(model_name)
    reshape_allowed = config.mxnet_models[model_name].get('reshape')
    shape = config.get_shape(model_name)
    if reshape_allowed is True and im_size is not None:
        shape = (1, 3) + tuple(im_size)[::-1]

    trt_dir, trt_path = config.build_model_paths(model_name, 'plan')

    if reshape_allowed is True:
        trt_path = trt_path.replace('.plan', f'_{shape[3]}_{shape[2]}.plan')
    if max_batch_size > 1:
        trt_path = trt_path.replace('.plan', f'_batch{max_batch_size}.plan')
    if force_fp16 is True:
        trt_path = trt_path.replace('.plan', '_fp16.plan')
    return trt_path


def get_model(model_name: str, im_size: List[int] = None, max_batch_size: int = 1,
              force_fp16: bool = False,
              root_dir: str = "/models", download_model: bool = True, **kwargs):
    """
    Returns inference backend instance with loaded model.

    :param model_name: Name of required model. Must be one of keys in `models` dict.
    :param backend_name: Name of inference backend. (onnx, mxnet, trt)
    :param im_size: Desired maximum size of image in W,H form. Will be overridden if model doesn't support reshaping.
    :param max_batch_size: Maximum batch size for inference, currently supported for ArcFace model only.
    :param force_fp16: Force use of FP16 precision, even if device doesn't support it. Be careful. TensorRT specific.
    :param root_dir: Root directory where models will be stored.
    :param download_model: Download MXNet or ONNX model. Might be disabled if TRT model was already created.
    :param kwargs: Placeholder.
    :return: Inference backend with loaded model.
    """

    config = Configs(models_dir=root_dir)
    model_path = prepare_backend(model_name, im_size=im_size, max_batch_size=max_batch_size,
                                 config=config, force_fp16=force_fp16,
                                 download_model=download_model)

    outputs = config.get_outputs_order(model_name)
    model = models[model_name](model_path=model_path, backend=trt_backend, outputs=outputs)
    return model
