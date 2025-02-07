import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import os
import torch
from collections import OrderedDict
from fvcore.common.file_io import PathManager
from tabulate import tabulate
import torchvision

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.structures import BoxMode
from detectron2.utils.logger import create_small_table

from fsdet.utils.file_io import PathManager
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from fsdet.evaluation.evaluator import DatasetEvaluator

class CarStickerEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name, cfg, distributed, output_dir=None):
        self._distributed = distributed
        self._output_dir = output_dir
        self._dataset_name = dataset_name

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        self._metadata = MetadataCatalog.get(dataset_name) 
        
        json_file = PathManager.get_local_path(self._metadata.json_file)
        self._json_file = json_file

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}
            if "instances" in output:
                pred_instances = output["instances"]
                
                # Get boxes, scores and classes on CPU
                boxes = pred_instances.pred_boxes.tensor.cpu().numpy()
                scores = pred_instances.scores.cpu().numpy()
                classes = pred_instances.pred_classes.cpu().numpy()
                for box, score, cls in zip(boxes, scores, classes):
                    # Convert box to COCO format (x, y, width, height)
                    x_min, y_min, x_max, y_max = box
                    width = x_max - x_min
                    height = y_max - y_min
                    coco_box = [float(x_min), float(y_min), float(width), float(height)]
                    
                    prediction_item = {
                        "image_id": input["image_id"],
                        "category_id": cls.item(), 
                        "bbox": coco_box,  
                        "score": float(score),  
                    }
                    self._predictions.append(prediction_item)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            self._predictions = comm.gather(self._predictions, dst=0)
            self._predictions = list(itertools.chain(*self._predictions))

            if not comm.is_main_process():
                return {}

        if len(self._predictions) == 0:
            self._logger.warning("[CarStickerEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "car_sticker_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(self._predictions, f)

        self._results = OrderedDict()
        self._eval_predictions()

        return copy.deepcopy(self._results)

    def _eval_predictions(self):
        self._prepare_annotations()
      
        # self._results.update(ap_result)
        # self._logger.info("Car sticker evaluation results: {}".format(self._results))

    def _prepare_annotations(self):
        gt_annotations = []
        pred_annotations = []
        processed_image_ids = set()  # Maintain a set to store processed image IDs

        # reverse ID map for 60 base classes + 1 novel (car-sticker), car-sticker is category-id = 91
        metadata = self._metadata
        print('DATASET_NAME:', self._dataset_name)
        print('JSON_FILE: ', self._json_file)
        
        is_tinyonly = "tinyonly" in self._dataset_name
        is_top4 = "top4" in self._dataset_name
        has_windshield = "ws" in self._dataset_name

        # Determine the correct mapping based on dataset name.
        if is_tinyonly and is_top4 and has_windshield:
            IDMAP = metadata.tinyonly_top4_stickers_ws_id_to_contiguous_id
        elif has_windshield:
            IDMAP = metadata.stickers_ws
        elif is_top4:
            IDMAP = metadata.tinyonly_top4_stickers_id_to_contiguous_id
        elif is_tinyonly:
            IDMAP = metadata.tinyonly_stickers_id_to_contiguous_id
        else:
            IDMAP = metadata.novel_dataset_id_to_contiguous_id
      
        inverse_IDMAP = {v: k for k, v in IDMAP.items()}

        print('**** INVERSE MAP TO BE USED IN EVALUATION *****')
        print(inverse_IDMAP)
        
        for prediction in self._predictions:
            if prediction["image_id"] not in processed_image_ids:
               
                processed_image_ids.add(prediction["image_id"])
            # Remap to original category id
            prediction['category_id'] = inverse_IDMAP[prediction['category_id']]
            pred_annotations.append(prediction)

        # Save predictions to a JSON file.
        predictions_json = json.dumps(pred_annotations)
        pred_json_path = "car_sticker_predictions.json"
        with open(pred_json_path, "w") as f:
            f.write(predictions_json)

        # Load ground truth annotations
        coco_gt = COCO(self._json_file)
        coco_gt.createIndex()
        coco_pred = coco_gt.loadRes(pred_json_path)

      
        # Overall evaluation (all categories)
  
        coco_eval_overall = COCOeval(coco_gt, coco_pred, 'bbox')
        coco_eval_overall.evaluate()
        coco_eval_overall.accumulate()
        overall_buffer = io.StringIO()
        with contextlib.redirect_stdout(overall_buffer):
            coco_eval_overall.summarize()
        overall_summary = overall_buffer.getvalue()
        print("OVERALL EVALUATION SUMMARY:")
        print(overall_summary)

        # Write the overall evaluation summary to a text file.
        overall_output_file = "coco_eval_results.txt"
        with open(overall_output_file, "w") as f:
            f.write("OVERALL EVALUATION SUMMARY:\n")
            f.write(overall_summary)

       
        # Per-class evaluation
      
        per_class_results = "PER-CLASS EVALUATION SUMMARY:\n\n"
        for catId in coco_gt.getCatIds():
            # Get cat name
            cat_info = coco_gt.loadCats([catId])[0]
            cat_name = cat_info['name']

            per_class_results += f"Category {catId} ({cat_name}):\n"
            coco_eval_cat = COCOeval(coco_gt, coco_pred, 'bbox')
            coco_eval_cat.params.catIds = [catId]
            coco_eval_cat.evaluate()
            coco_eval_cat.accumulate()
            cat_buffer = io.StringIO()
            with contextlib.redirect_stdout(cat_buffer):
                coco_eval_cat.summarize()
            cat_summary = cat_buffer.getvalue()
            per_class_results += cat_summary + "\n\n"

        print(per_class_results)

        # Write the per-class evaluation summary to a separate text file
        per_class_output_file = "coco_eval_per_class_results.txt"
        with open(per_class_output_file, "w") as f:
            f.write(per_class_results)

      
        self._results["overall"] = overall_summary
        self._results["per_class"] = per_class_results

def _evaluate_predictions_on_coco(
        coco_gt, coco_results, iou_type, catIds=None
    ):
    """
    Evaluate the coco results using COCOEval API.
    """
    assert len(coco_results) > 0
    
    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    if catIds is not None:
        coco_eval.params.catIds = catIds
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
    return coco_eval
