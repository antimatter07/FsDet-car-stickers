import contextlib
import copy
import io
import itertools
import json
import logging
import os
from collections import OrderedDict

import detectron2.utils.comm as comm
import numpy as np
import torch
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.structures import BoxMode
from detectron2.utils.logger import create_small_table
from fsdet.evaluation.evaluator import DatasetEvaluator
from fsdet.utils.file_io import PathManager
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

#
inverse_IDMAP = {
    0: 1, 1: 8, 2: 10, 3: 11, 4: 13, 5: 14, 6: 15, 7: 22, 8: 23, 9: 24, 10: 25,
    11: 27, 12: 28, 13: 31, 14: 32, 15: 33, 16: 34, 17: 35, 18: 36, 19: 37, 20: 38,
    21: 39, 22: 40, 23: 41, 24: 42, 25: 43, 26: 46, 27: 47, 28: 48, 29: 49, 30: 50,
    31: 51, 32: 52, 33: 53, 34: 54, 35: 55, 36: 56, 37: 57, 38: 58, 39: 59, 40: 60,
    41: 61, 42: 65, 43: 70, 44: 73, 45: 74, 46: 75, 47: 76, 48: 77, 49: 78, 50: 79,
    51: 80, 52: 81, 53: 82, 54: 84, 55: 85, 56: 86, 57: 87, 58: 88, 59: 89, 60: 90
}

predictions = load_json()