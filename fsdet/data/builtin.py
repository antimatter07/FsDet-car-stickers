"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.
We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations
We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".

Here we only register the few-shot datasets and complete COCO, PascalVOC and
LVIS have been handled by the builtin datasets in detectron2.
"""

import os

from detectron2.data import MetadataCatalog
from detectron2.data.datasets.lvis import register_lvis_instances

from .builtin_meta import _get_builtin_metadata
from .meta_coco import register_meta_coco
from .meta_lvis import register_meta_lvis
from .meta_pascal_voc import register_meta_pascal_voc

from .meta_stickers import register_meta_stickers

# ==== Predefined datasets and splits for COCO ==========

_PREDEFINED_SPLITS_COCO = {}
_PREDEFINED_SPLITS_COCO["coco"] = {
    "coco_2014_train": (
        "coco/train2014",
        "coco/annotations/instances_train2014.json",
    ),
    "coco_2014_val": (
        "coco/val2014",
        "coco/annotations/instances_val2014.json",
    ),
    "coco_2014_minival": (
        "coco/val2014",
        "coco/annotations/instances_minival2014.json",
    ),
    "coco_2014_minival_100": (
        "coco/val2014",
        "coco/annotations/instances_minival2014_100.json",
    ),
    "coco_2014_valminusminival": (
        "coco/val2014",
        "coco/annotations/instances_valminusminival2014.json",
    ),
    "coco_2017_train": (
        "coco/train2017",
        "coco/annotations/instances_train2017.json",
    ),
    "coco_2017_val": (
        "coco/val2017",
        "coco/annotations/instances_val2017.json",
    ),
    "coco_2017_test": (
        "coco/test2017",
        "coco/annotations/image_info_test2017.json",
    ),
    "coco_2017_test-dev": (
        "coco/test2017",
        "coco/annotations/image_info_test-dev2017.json",
    ),
    "coco_2017_val_100": (
        "coco/val2017",
        "coco/annotations/instances_val2017_100.json",
    ),
}


def register_all_coco(root="datasets"):
    # for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_COCO.items():
    #     for key, (image_root, json_file) in splits_per_dataset.items():
    #         # Assume pre-defined datasets live in `./datasets`.
    #         register_coco_instances(
    #             key,
    #             _get_builtin_metadata(dataset_name),
    #             os.path.join(root, json_file)
    #             if "://" not in json_file
    #             else json_file,
    #             os.path.join(root, image_root),
    #         )

    # register meta datasets
    METASPLITS = [
        (
            "coco_trainval_all",
            "coco/trainval2014",
            "cocosplit/datasplit/trainvalno5k.json",
        ),
        (
            "coco_trainval_base",
            "coco/trainval2014",
            "cocosplit/datasplit/trainvalno5k.json",
        ),
        ("coco_test_all", "coco/val2014", "cocosplit/datasplit/5k.json"),
        ("coco_test_base", "coco/val2014", "cocosplit/datasplit/5k.json"),
        ("coco_test_novel", "coco/val2014", "cocosplit/datasplit/5k.json"),
    ]

    # register small meta datasets for fine-tuning stage
    for prefix in ["all", "novel"]:
        for shot in [1, 2, 3, 5, 10, 30]:
            for seed in range(10):
                seed = "" if seed == 0 else "_seed{}".format(seed)
                name = "coco_trainval_{}_{}shot{}".format(prefix, shot, seed)
                METASPLITS.append((name, "coco/trainval2014", ""))
    #ADDED BELOW 
    # register datasets for X amount of shots from base classes
    for shot in [1, 2, 3, 5, 10, 30]:
        name = "coco_trainval_base_{}shot".format(shot)
        METASPLITS.append((name, "coco/trainval2014", ""))

    for name, imgdir, annofile in METASPLITS:
        register_meta_coco(
            name,
            _get_builtin_metadata("coco_fewshot"),
            os.path.join(root, imgdir),
            os.path.join(root, annofile),
        )


# ==== Predefined datasets and splits for LVIS ==========

_PREDEFINED_SPLITS_LVIS = {
    "lvis_v0.5": {
        # "lvis_v0.5_train": ("coco/train2017", "lvis/lvis_v0.5_train.json"),
        "lvis_v0.5_train_freq": (
            "coco/train2017",
            "lvis/lvis_v0.5_train_freq.json",
        ),
        "lvis_v0.5_train_common": (
            "coco/train2017",
            "lvis/lvis_v0.5_train_common.json",
        ),
        "lvis_v0.5_train_rare": (
            "coco/train2017",
            "lvis/lvis_v0.5_train_rare.json",
        ),
        # "lvis_v0.5_val": ("coco/val2017", "lvis/lvis_v0.5_val.json"),
        # "lvis_v0.5_val_rand_100": (
        #     "coco/val2017",
        #     "lvis/lvis_v0.5_val_rand_100.json",
        # ),
        # "lvis_v0.5_test": (
        #     "coco/test2017",
        #     "lvis/lvis_v0.5_image_info_test.json",
        # ),
    },
}


def register_all_lvis(root="datasets"):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_LVIS.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_lvis_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file)
                if "://" not in json_file
                else json_file,
                os.path.join(root, image_root),
            )

    # register meta datasets
    METASPLITS = [
        (
            "lvis_v0.5_train_shots",
            "coco/train2017",
            "lvissplit/lvis_shots.json",
        ),
        (
            "lvis_v0.5_train_rare_novel",
            "coco/train2017",
            "lvis/lvis_v0.5_train_rare.json",
        ),
        ("lvis_v0.5_val_novel", "coco/val2017", "lvis/lvis_v0.5_val.json"),
    ]

    for name, image_root, json_file in METASPLITS:
        dataset_name = "lvis_v0.5_fewshot" if "novel" in name else "lvis_v0.5"
        register_meta_lvis(
            name,
            _get_builtin_metadata(dataset_name),
            os.path.join(root, json_file)
            if "://" not in json_file
            else json_file,
            os.path.join(root, image_root),
        )


# ==== Predefined splits for PASCAL VOC ===========
def register_all_pascal_voc(root="datasets"):
    # SPLITS = [
    #     ("voc_2007_trainval", "VOC2007", "trainval"),
    #     ("voc_2007_train", "VOC2007", "train"),
    #     ("voc_2007_val", "VOC2007", "val"),
    #     ("voc_2007_test", "VOC2007", "test"),
    #     ("voc_2012_trainval", "VOC2012", "trainval"),
    #     ("voc_2012_train", "VOC2012", "train"),
    #     ("voc_2012_val", "VOC2012", "val"),
    # ]
    # for name, dirname, split in SPLITS:
    #     year = 2007 if "2007" in name else 2012
    #     register_pascal_voc(name, os.path.join(root, dirname), split, year)
    #     MetadataCatalog.get(name).evaluator_type = "pascal_voc"

    # register meta datasets
    METASPLITS = [
        ("voc_2007_trainval_base1", "VOC2007", "trainval", "base1", 1),
        ("voc_2007_trainval_base2", "VOC2007", "trainval", "base2", 2),
        ("voc_2007_trainval_base3", "VOC2007", "trainval", "base3", 3),
        ("voc_2012_trainval_base1", "VOC2012", "trainval", "base1", 1),
        ("voc_2012_trainval_base2", "VOC2012", "trainval", "base2", 2),
        ("voc_2012_trainval_base3", "VOC2012", "trainval", "base3", 3),
        ("voc_2007_trainval_all1", "VOC2007", "trainval", "base_novel_1", 1),
        ("voc_2007_trainval_all2", "VOC2007", "trainval", "base_novel_2", 2),
        ("voc_2007_trainval_all3", "VOC2007", "trainval", "base_novel_3", 3),
        ("voc_2012_trainval_all1", "VOC2012", "trainval", "base_novel_1", 1),
        ("voc_2012_trainval_all2", "VOC2012", "trainval", "base_novel_2", 2),
        ("voc_2012_trainval_all3", "VOC2012", "trainval", "base_novel_3", 3),
        ("voc_2007_test_base1", "VOC2007", "test", "base1", 1),
        ("voc_2007_test_base2", "VOC2007", "test", "base2", 2),
        ("voc_2007_test_base3", "VOC2007", "test", "base3", 3),
        ("voc_2007_test_novel1", "VOC2007", "test", "novel1", 1),
        ("voc_2007_test_novel2", "VOC2007", "test", "novel2", 2),
        ("voc_2007_test_novel3", "VOC2007", "test", "novel3", 3),
        ("voc_2007_test_all1", "VOC2007", "test", "base_novel_1", 1),
        ("voc_2007_test_all2", "VOC2007", "test", "base_novel_2", 2),
        ("voc_2007_test_all3", "VOC2007", "test", "base_novel_3", 3),
    ]

    # register small meta datasets for fine-tuning stage
    for prefix in ["all", "novel"]:
        for sid in range(1, 4):
            for shot in [1, 2, 3, 5, 10]:
                for year in [2007, 2012]:
                    for seed in range(100):
                        seed = "" if seed == 0 else "_seed{}".format(seed)
                        name = "voc_{}_trainval_{}{}_{}shot{}".format(
                            year, prefix, sid, shot, seed
                        )
                        dirname = "VOC{}".format(year)
                        img_file = "{}_{}shot_split_{}_trainval".format(
                            prefix, shot, sid
                        )
                        keepclasses = (
                            "base_novel_{}".format(sid)
                            if prefix == "all"
                            else "novel{}".format(sid)
                        )
                        METASPLITS.append(
                            (name, dirname, img_file, keepclasses, sid)
                        )

    for name, dirname, split, keepclasses, sid in METASPLITS:
        year = 2007 if "2007" in name else 2012
        register_meta_pascal_voc(
            name,
            _get_builtin_metadata("pascal_voc_fewshot"),
            os.path.join(root, dirname),
            split,
            year,
            keepclasses,
            sid,
        )
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"


# Register them all under "./datasets"
register_all_coco()
register_all_lvis()
register_all_pascal_voc()
#def register_meta_stickers("datasets/stickers/", image_root, metadata, dataset_name)
#stickers fewshot - stickers_28shot
register_meta_stickers("datasets/stickers_split/full_box_28shot_train.json", "datasets/stickers/stickers_train_28shot", _get_builtin_metadata("stickers_fewshot"), "stickers_fewshot")
register_meta_stickers("datasets/stickers_split/full_box_28shot_train.json", "datasets/stickers/stickers_train_28shot", _get_builtin_metadata("stickers_fewshot"), "tinyonly_stickers_28shot")
register_meta_stickers("datasets/stickers_split/full_box_28shot_train.json", "datasets/stickers/stickers_train_28shot", _get_builtin_metadata("stickers_fewshot"), "tinyonly_top4_stickers_28shot")


#31 shot

register_meta_stickers("datasets/stickers_split/stickers_train_31shot_1400.json", "datasets/stickers/stickers_train_31shot_1400", _get_builtin_metadata("stickers_fewshot"), "stickers_31shot_1400")
register_meta_stickers("datasets/stickers_split/stickers_train_31shot_1400.json", "datasets/stickers/stickers_train_31shot_1400", _get_builtin_metadata("stickers_fewshot"), "tinyonly_stickers_31shot_1400")
register_meta_stickers("datasets/stickers_split/stickers_train_31shot_1400.json", "datasets/stickers/stickers_train_31shot_1400", _get_builtin_metadata("stickers_fewshot"), "tinyonly_top4_stickers_31shot_1400")


#TEST SETS STICKERS
register_meta_stickers("datasets/stickers/annotations/stickers_28shot_test.json", "datasets/stickers/stickers_test_28shot", _get_builtin_metadata("stickers_fewshot"), "stickers_28shot_test")
register_meta_stickers("datasets/stickers/annotations/stickers_31shot_test.json", "datasets/stickers/stickers_test_31shot_140", _get_builtin_metadata("stickers_fewshot"), "stickers_31shot_1400_test")
register_meta_stickers("datasets/stickers/annotations/stickers_31shot_test_1280.json", "datasets/stickers/stickers_all", _get_builtin_metadata("stickers_fewshot"), "stickers_31shot_1280_test")

#top4 version (for logic in stickers evaluator for correct ID mapping, content is exactly the same as "stickers_31shot_1280_test" above)
# use in YAML file for experiments using top 4 tiny only classes 
register_meta_stickers("datasets/stickers/annotations/stickers_31shot_test_1280.json", "datasets/stickers/stickers_all", _get_builtin_metadata("stickers_fewshot"), "stickers_31shot_1280_test_tinyonly")

register_meta_stickers("datasets/stickers/annotations/stickers_31shot_test_1280.json", "datasets/stickers/stickers_all", _get_builtin_metadata("stickers_fewshot"), "stickers_31shot_1280_test_tinyonly_top4")


#COCO dataset
register_meta_coco(
            "30_tinyonly",
            _get_builtin_metadata("tinyonly_fewshot"),
            os.path.join("datasets/coco/trainval2014"),
            os.path.join("datasets/cocosplit/30shot_tinyonly.json"),
        )
register_meta_coco("30_tinyonly_top4",
                  _get_builtin_metadata("tinyonly_top4_fewshot"),
                  os.path.join("datasets/coco/trainval2014"),
                  os.path.join("datasets/cocosplit/top4_tinyonly.json"),
                  )
#31shot 1280
register_meta_stickers("datasets/stickers_split/stickers_train_31shot_1280.json", "datasets/stickers/stickers_all", _get_builtin_metadata("stickers_fewshot"), "stickers_31shot_1280")
register_meta_stickers("datasets/stickers_split/stickers_train_31shot_1280.json", "datasets/stickers/stickers_all", _get_builtin_metadata("stickers_fewshot"), "tinyonly_stickers_31shot_1280")
register_meta_stickers("datasets/stickers_split/stickers_train_31shot_1280.json", "datasets/stickers/stickers_all", _get_builtin_metadata("stickers_fewshot"), "tinyonly_top4_stickers_31shot_1280")

#31shot 1280 with windshield class and annotations
register_meta_stickers("datasets/stickers_split/stickers_ws_train_31shot_1280.json", "datasets/stickers/stickers_ws_all", _get_builtin_metadata("stickers_fewshot"), "tinyonly_top4_stickers_ws_31shot_1280")

register_meta_stickers("datasets/stickers/annotations/stickers_ws_31shot_test_1280.json", "datasets/stickers/stickers_ws_all", _get_builtin_metadata("stickers_fewshot"), "stickers_ws_31shot_1280_test_tinyonly_top4")

# 1 shot
register_meta_stickers("datasets/stickers_split/stickers_ws_train_1shot_1280.json", "datasets/stickers/stickers_ws_all", _get_builtin_metadata("stickers_fewshot"), "tinyonly_top4_stickers_ws_1shot_1280")

register_meta_stickers("datasets/stickers/annotations/stickers_ws_1shot_test_1280.json", "datasets/stickers/stickers_ws_all", _get_builtin_metadata("stickers_fewshot"), "stickers_ws_1shot_1280_test_tinyonly_top4")

# 2 shot
register_meta_stickers("datasets/stickers_split/stickers_ws_train_2shot_1280.json", "datasets/stickers/stickers_ws_all", _get_builtin_metadata("stickers_fewshot"), "tinyonly_top4_stickers_ws_2shot_1280")

register_meta_stickers("datasets/stickers/annotations/stickers_ws_2shot_test_1280.json", "datasets/stickers/stickers_ws_all", _get_builtin_metadata("stickers_fewshot"), "stickers_ws_2shot_1280_test_tinyonly_top4")

# 3 shot

register_meta_stickers("datasets/stickers_split/stickers_ws_train_3shot_1280.json", "datasets/stickers/stickers_ws_all", _get_builtin_metadata("stickers_fewshot"), "tinyonly_top4_stickers_ws_3shot_1280")

register_meta_stickers("datasets/stickers/annotations/stickers_ws_3shot_test_1280.json", "datasets/stickers/stickers_ws_all", _get_builtin_metadata("stickers_fewshot"), "stickers_ws_3shot_1280_test_tinyonly_top4")

# 5 shot

register_meta_stickers("datasets/stickers_split/stickers_ws_train_5shot_1280.json", "datasets/stickers/stickers_ws_all", _get_builtin_metadata("stickers_fewshot"), "tinyonly_top4_stickers_ws_5shot_1280")

register_meta_stickers("datasets/stickers/annotations/stickers_ws_5shot_test_1280.json", "datasets/stickers/stickers_ws_all", _get_builtin_metadata("stickers_fewshot"), "stickers_ws_5shot_1280_test_tinyonly_top4")

# 10 shot

register_meta_stickers("datasets/stickers_split/stickers_ws_train_10shot_1280.json", "datasets/stickers/stickers_ws_all", _get_builtin_metadata("stickers_fewshot"), "tinyonly_top4_stickers_ws_10shot_1280")

register_meta_stickers("datasets/stickers/annotations/stickers_ws_10shot_test_1280.json", "datasets/stickers/stickers_ws_all", _get_builtin_metadata("stickers_fewshot"), "stickers_ws_10shot_1280_test_tinyonly_top4")

# STICKERS ONLY

# 1 shot
register_meta_stickers("datasets/stickers_split/stickers_train_1shot_1280.json", "datasets/stickers/stickers_all", _get_builtin_metadata("stickers_fewshot"), "tinyonly_top4_stickers_1shot_1280")

register_meta_stickers("datasets/stickers/annotations/stickers_1shot_test_1280.json", "datasets/stickers/stickers_all", _get_builtin_metadata("stickers_fewshot"), "stickers_1shot_1280_test_tinyonly_top4")

# 2 shot
register_meta_stickers("datasets/stickers_split/stickers_train_2shot_1280.json", "datasets/stickers/stickers_all", _get_builtin_metadata("stickers_fewshot"), "tinyonly_top4_stickers_2shot_1280")

register_meta_stickers("datasets/stickers/annotations/stickers_2shot_test_1280.json", "datasets/stickers/stickers_all", _get_builtin_metadata("stickers_fewshot"), "stickers_2shot_1280_test_tinyonly_top4")

# 3 shot

register_meta_stickers("datasets/stickers_split/stickers_train_3shot_1280.json", "datasets/stickers/stickers_all", _get_builtin_metadata("stickers_fewshot"), "tinyonly_top4_stickers_3shot_1280")

register_meta_stickers("datasets/stickers/annotations/stickers_3shot_test_1280.json", "datasets/stickers/stickers_all", _get_builtin_metadata("stickers_fewshot"), "stickers_3shot_1280_test_tinyonly_top4")

# 5 shot

register_meta_stickers("datasets/stickers_split/stickers_train_5shot_1280.json", "datasets/stickers/stickers_all", _get_builtin_metadata("stickers_fewshot"), "tinyonly_top4_stickers_5shot_1280")

register_meta_stickers("datasets/stickers/annotations/stickers_5shot_test_1280.json", "datasets/stickers/stickers_all", _get_builtin_metadata("stickers_fewshot"), "stickers_5shot_1280_test_tinyonly_top4")

# 10 shot

register_meta_stickers("datasets/stickers_split/stickers_train_10shot_1280.json", "datasets/stickers/stickers_all", _get_builtin_metadata("stickers_fewshot"), "tinyonly_top4_stickers_10shot_1280")

register_meta_stickers("datasets/stickers/annotations/stickers_10shot_test_1280.json", "datasets/stickers/stickers_all", _get_builtin_metadata("stickers_fewshot"), "stickers_10shot_1280_test_tinyonly_top4")


