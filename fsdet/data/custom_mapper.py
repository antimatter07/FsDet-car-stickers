from detectron2.data import detection_utils as utils
from detectron2.data import DatasetMapper
import torch
import numpy as np
import cv2

class TwoStepMapper(DatasetMapper):
    def __init__(self, cfg, ws_model, is_train=True):
        self.is_train = is_train
        self.augmentations = utils.build_augmentation(cfg, is_train)
        self.image_format = cfg.INPUT.FORMAT

    def __call__(self, dataset_dict):
        # Load image
        dataset_dict = dataset_dict.copy()
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)

        # Detect windshields