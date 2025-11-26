"""
Utilities for loading COCO-format annotations into Detectron2-format
dataset_dicts, including support for few-shot splits, base-only splits,
novel-only splits, and tiny-object filtering for generating tiny only COCO split.
"""


import contextlib
import io
import os
# added import json
import json

import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from fsdet.utils.file_io import PathManager
from pycocotools.coco import COCO



__all__ = ["register_meta_coco"]


def load_coco_json(json_file, image_root, metadata, dataset_name):
    """
    Load a json file with COCO's instances annotation format.
    Currently supports instance detection.
    Args:
        json_file (str): full path to the json file in COCO instances annotation format.
        image_root (str): the directory where the images in this json file exists.
        metadata: meta data associated with dataset_name
        dataset_name (str): the name of the dataset (e.g., coco_2017_train).
            If provided, this function will also put "thing_classes" into
            the metadata associated with this dataset.
    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """

    #ADDED CHANGES FOR BASE SHOTS
    is_shots = "shot" in dataset_name
    
    is_base = "base" in dataset_name
    is_tinyonly = "tinyonly" in dataset_name

    tiny_object_count = 0

    #print('**BASE CLASSES!!!**')
    #print(metadata["base_classes"])

    tinyonly_dataset = {
        "info": [],
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []

        
    }
    
    if is_shots:
        fileids = {}
        split_dir = os.path.join("datasets", "cocosplit")
        if "seed" in dataset_name:
            shot = dataset_name.split("_")[-2].split("shot")[0]
            seed = int(dataset_name.split("_seed")[-1])
            split_dir = os.path.join(split_dir, "seed{}".format(seed))
        else:
            shot = dataset_name.split("_")[-1].split("shot")[0]

        if is_base:
            print('*********IN BASE SHOTS************')
            for idx, cls in enumerate(metadata["base_classes"]):
                json_file = os.path.join(
                    split_dir, "full_box_{}shot_{}_trainval.json".format(shot, cls)
                )
                json_file = PathManager.get_local_path(json_file)
                with contextlib.redirect_stdout(io.StringIO()):
                    coco_api = COCO(json_file)

                # ADDED BELOW FOR MAKING TINY ONLY DATASET 
                average_sticker_area = 205.15810674723062
                
                with open(json_file, 'r') as f:
                    data = json.load(f)

                tiny_anns = []

                #filter out annotations larger than the average 
                for anno in data['annotations']:
                    if anno['area'] <= average_sticker_area:
                        tiny_anns.append(anno)
                        tiny_object_count += 1

                tinyonly_dataset['images'].extend(data['images'])
                tinyonly_dataset['annotations'].extend(tiny_anns)
                tinyonly_dataset['categories'] = data['categories']
                
                    


                #ADDED ABOVE FOR MAKING TINY ONLY DATASET
                
                img_ids = sorted(list(coco_api.imgs.keys()))
                imgs = coco_api.loadImgs(img_ids)
                anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
                fileids[idx] = list(zip(imgs, anns))

        else:
            for idx, cls in enumerate(metadata["thing_classes"]):
                json_file = os.path.join(
                    split_dir, "full_box_{}shot_{}_trainval.json".format(shot, cls)
                )
                json_file = PathManager.get_local_path(json_file)
                with contextlib.redirect_stdout(io.StringIO()):
                    coco_api = COCO(json_file)
                img_ids = sorted(list(coco_api.imgs.keys()))
                imgs = coco_api.loadImgs(img_ids)
                anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
                fileids[idx] = list(zip(imgs, anns))
    else:
        json_file = PathManager.get_local_path(json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            coco_api = COCO(json_file)
        # sort indices for reproducible results
        img_ids = sorted(list(coco_api.imgs.keys()))
        imgs = coco_api.loadImgs(img_ids)
        anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
        imgs_anns = list(zip(imgs, anns))

    if is_tinyonly:
        id_map = metadata["tinyonly_id_to_contiguous_id"]
    else:
        id_map = metadata["thing_dataset_id_to_contiguous_id"]

    print('****** TINY OBJECT COUNT ********')
    print(tiny_object_count)

    #ADDED BELOW FOR TRACING
    print('****PRINTING ID_MAP IN META_COCO.PY*****')
    print(id_map)

    

    
    dataset_dicts = []
    ann_keys = ["iscrowd", "bbox", "category_id"]

    if is_shots:
        for _, fileids_ in fileids.items():
            dicts = []
            for (img_dict, anno_dict_list) in fileids_:
                for anno in anno_dict_list:
                    record = {}
                    record["file_name"] = os.path.join(
                        image_root, img_dict["file_name"]
                    )
                    record["height"] = img_dict["height"]
                    record["width"] = img_dict["width"]
                    image_id = record["image_id"] = img_dict["id"]

                    assert anno["image_id"] == image_id
                    assert anno.get("ignore", 0) == 0

                    obj = {key: anno[key] for key in ann_keys if key in anno}

                    obj["bbox_mode"] = BoxMode.XYWH_ABS
                    obj["category_id"] = id_map[obj["category_id"]]
                    record["annotations"] = [obj]
                    dicts.append(record)
            if len(dicts) > int(shot):
                dicts = np.random.choice(dicts, int(shot), replace=False)
            dataset_dicts.extend(dicts)
    else:
        for (img_dict, anno_dict_list) in imgs_anns:
            record = {}
            record["file_name"] = os.path.join(
                image_root, img_dict["file_name"]
            )
            record["height"] = img_dict["height"]
            record["width"] = img_dict["width"]
            image_id = record["image_id"] = img_dict["id"]

            objs = []
            for anno in anno_dict_list:
                assert anno["image_id"] == image_id
                assert anno.get("ignore", 0) == 0

                obj = {key: anno[key] for key in ann_keys if key in anno}

                obj["bbox_mode"] = BoxMode.XYWH_ABS
                if obj["category_id"] in id_map:
                    obj["category_id"] = id_map[obj["category_id"]]
                    objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)

    #ADDED BELOW FOR TRACING
    records_json = json.dumps(dataset_dicts)

    fpath = "res.json"
    with open(fpath, "w") as f:
        f.write(records_json)
    #LINES BELOW ADDED FOR SAVING TINY ONLY DATASET
    #tinyonly_dataset_json = json.dumps(tinyonly_dataset)

    #comment below if no need to save dataset
    #file_path_tiny = "{}shot_tiny_dataset.json".format(shot)
    #with open(file_path_tiny, "w") as f:
    #        f.write(tinyonly_dataset_json)

    #LINES ABOVE ADDED FOR SAVING TINY ONLY DATASET
    return dataset_dicts


def register_meta_coco(name, metadata, imgdir, annofile):
    """
    Register COCO or COCO-style dataset into Detectron2.

    Handles:
    -Standard COCO dataset  
    -Base-only / Novel-only splits (dataset_name contains "_base" or "_novel")  
    -Metadata injection (categories, id mappings, paths, evaluator type)  

    Args:
        name (str):
            Dataset name used in DatasetCatalog.
        metadata (dict):
            Should contain:
                - base_classes / novel_classes
                - *_dataset_id_to_contiguous_id
                - thing_classes
        imgdir (str):
            Directory containing images.
        annofile (str):
            Path to annotation JSON.

    Returns:
        None
    """
    DatasetCatalog.register(
        name,
        lambda: load_coco_json(annofile, imgdir, metadata, name),
    )

    if "_base" in name or "_novel" in name:
        split = "base" if "_base" in name else "novel"
        metadata["thing_dataset_id_to_contiguous_id"] = metadata[
            "{}_dataset_id_to_contiguous_id".format(split)
        ]
        metadata["thing_classes"] = metadata["{}_classes".format(split)]

    MetadataCatalog.get(name).set(
        json_file=annofile,
        image_root=imgdir,
        evaluator_type="coco",
        dirname="datasets/coco",
        **metadata,
    )
