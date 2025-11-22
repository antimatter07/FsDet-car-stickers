
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog


import io
import contextlib
import os
import numpy as np
import json

#from fsdet.data import DatasetCatalog, MetadataCatalog
#from fsdet.structures import BoxMode
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO

def load_stickers_json(json_file, image_root, metadata, dataset_name):
    """
    Load a COCO-format stickers dataset and convert it into Detectron2 dataset_dicts.

    Args:
        json_file (str): Path to the COCO JSON annotation file.
        image_root (str): Directory where the images are stored.
        metadata (dict): Metadata dictionary containing ID mappings and class info.
        dataset_name (str): Dataset name used to determine which IDMAP rules apply.

    Behavior:
        - Supports multiple dataset modes like:
            * `tinyonly`
            * `top4`
            * `ws` (with windshield)
            * `stickers_only`
        - Automatically selects the correct ID-to-contiguous-ID mapping based on naming.
        - Converts COCO annotations into Detectron2 records:
            file_name, height, width, annotations (with bbox + category_id), image_id.

    Returns:
        list[dict]: A Detectron2 dataset_dict list for training/inference.
    """

    #if tinyonly config, data is loaded differently 
    is_tinyonly = "tinyonly" in dataset_name

    is_top4 = "top4" in dataset_name
    has_windshield = "ws" in dataset_name

    stickers_only = "stickers_only" in dataset_name
    
    with open(json_file, "r") as f:
        dataset_info = json.load(f)

    images = dataset_info["images"]
    annotations = dataset_info["annotations"]
    categories = dataset_info["categories"]

    id_to_category = {category["id"]: category["name"] for category in categories}
    thing_classes = [category["name"] for category in categories]

    # id map
    # in experiments, 91 maps to 60 in contiguous ID MAP whwere 0-59 is mapped to the 60 COCO base classes defined in FsDet
    # 91 is default sticker, appended as part of 'object' classes in COCO
    # Make sure to adjust IDMAP if needed 
    #IDMAP = {91 : 60}

    # "novel_dataset_id_to_contiguous_id", "tinyonly_stickers_id_to_contiguous_id"

    # tinyonly_top4_stickers_id_to_contiguous_id
    #print("DATASET STICKERS: " + dataset_name)
    if stickers_only:
        IDMAP = metadata["stickers_only"]
    elif is_tinyonly and is_top4 and has_windshield:
        IDMAP = metadata["tinyonly_top4_stickers_ws_id_to_contiguous_id"]
    elif has_windshield:
        IDMAP = metadata["stickers_ws"]
    elif is_top4:
        IDMAP = metadata["tinyonly_top4_stickers_id_to_contiguous_id"]
    elif is_tinyonly:
        IDMAP = metadata["tinyonly_stickers_id_to_contiguous_id"]
    else:
        IDMAP = metadata["novel_dataset_id_to_contiguous_id"]

    #print('***IDMAP in meta_stickers.py**')
    #print(IDMAP)

    dataset_dicts = []
    for image in images:
        record = {}
        record["file_name"] = os.path.join(image_root, image["file_name"])
        record["image_id"] = image["id"]
        record["height"] = image["height"]
        record["width"] = image["width"]

        objs = []
        for annotation in annotations:
            if annotation["image_id"] == image["id"]:
                obj = {
                    "iscrowd": 0,
                    "bbox": annotation["bbox"],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "category_id": IDMAP[annotation["category_id"]],
                }
                objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)

    #print("DATASET MADE BY meta_stickers.py:")
    #print(dataset_dicts)

    return dataset_dicts

# Replace the following paths and dataset_name with your actual values
#json_file = "path/to/your/dataset.json"
#image_root = "path/to/your/images"
#dataset_name = "your_dataset_name"

# Register the custom dataset
#MetadataCatalog.get(dataset_name).set(thing_classes=thing_classes)
#DatasetCatalog.register(dataset_name, lambda: load_custom_dataset(json_file, image_root, MetadataCatalog.get(dataset_name), dataset_name))

# Usage example:
# dataset_dicts = DatasetCatalog.get(dataset_name)()

def register_meta_stickers(json_file, image_root, metadata, dataset_name):
    """
    Register the stickers dataset into Detectron2’s DatasetCatalog and MetadataCatalog.

    Args:
        json_file (str): Path to annotations JSON.
        image_root (str): Path to image folder.
        metadata (dict): Metadata containing class names, IDMAPs, etc.
        dataset_name (str): Dataset identifier.

    Behavior:
        - Step 1: Register dataset loader under DatasetCatalog.
        - Step 2: Attach metadata (IDMAPs, evaluator type, json paths, etc.)
        - Used inside config via cfg.DATASETS.TRAIN / TEST.

    The registered dataset can be loaded using:
        DatasetCatalog.get(dataset_name)
    """
    
    # register dataset (step 1)
    DatasetCatalog.register(
        dataset_name, # name of dataset, this will be used in the config file
        lambda: load_stickers_json( # this calls your dataset loader to get the data
            json_file, image_root, metadata, dataset_name # inputs to your dataset loader
        ),
    )
# json_file, image_root, metadata, dataset_name
    # register meta information (step 2)
    #MetadataCatalog.get(dataset_name).set(
    #    novel_classes=metadata["novel_classes"], # novel classes
    #)
    #MetadataCatalog.get(dataset_name).evaluator_type = "stickers" # set evaluator
    MetadataCatalog.get(dataset_name).set(
        json_file=json_file,
        image_root=image_root,
        evaluator_type="stickers",
        dirname="datasets/stickers",
        **metadata,
    )

 

