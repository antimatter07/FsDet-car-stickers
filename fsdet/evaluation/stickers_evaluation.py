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

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}
            if "instances" in output:
                pred_instances = output["instances"]
                
                #CPU version of line below : boxes = pred_instances.pred_boxes.tensor.cpu().numpy()
                boxes = pred_instances.pred_boxes.tensor.cpu().numpy()
                
                #scores = pred_instances.scores.numpy()
                scores = pred_instances.scores.cpu().numpy()
                classes = pred_instances.pred_classes.cpu().numpy()
                for box, score, cls in zip(boxes, scores, classes):
                    # convert box to COCO format (x, y, width, height)
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
                    #print('processed instance: ',prediction_item)
                    self._predictions.append(prediction_item)
            #print("Processed prediction:", prediction)

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
        gt_dataset, pred_annotations = self._prepare_annotations()

        
    
        self._results.update(ap_result)
        self._logger.info("Car sticker evaluation results: {}".format(self._results))

    

    def _prepare_annotations(self):
        gt_annotations = []
        pred_annotations = []
        processed_image_ids = set()  # Maintain a set to store processed image IDs

        # reverse ID map for 60 base classes + 1 novel (car-sticker), car-sticker is category-id = 91
        metadata = self._metadata
        print('DATASET_NAME:', self._dataset_name)
        is_tinyonly = "tinyonly" in self._dataset_name
        is_top4 = "top4" in self._dataset_name

        if is_top4:
            IDMAP = metadata.tinyonly_top4_stickers_id_to_contiguous_id
        elif is_tinyonly:
            IDMAP = metadata.tinyonly_stickers_id_to_contiguous_id
        else:
            IDMAP = metadata.novel_dataset_id_to_contiguous_id
      
        
      #  if is_tiny_only:
      #      IDMAP = metadata.tinyonly_stickers_id_to_contiguous_id
       # else:
       #     IDMAP = metadata.novel_dataset_id_to_contiguous_id

        #FOR NOW HANDLE CORRECT MAPPINGS FOR EACH DATASET BY MANUALLY CHANGING LINE BELOW
        #  "tinyonly_top4_stickers_id_to_contiguous_id"
        
        #IDMAP = metadata.tinyonly_stickers_id_to_contiguous_id
        #IDMAP = metadata.tinyonly_top4_stickers_id_to_contiguous_id

       

        
        
        inverse_IDMAP = {v: k for k, v in IDMAP.items()}

        print('**** INVERSE MAP TO BE USED IN EVALUATION *****')
        print(inverse_IDMAP)
        

        for prediction in self._predictions:

            if prediction["image_id"] not in processed_image_ids:
                # Add ground truth annotations only if the image ID hasn't been processed yet
                #gt_annotations.extend(get_ground_truth_annotations(prediction["image_id"]))
                processed_image_ids.add(prediction["image_id"])  # Add the image ID to the set of processed IDs
            # remap to original category id
            prediction['category_id'] = inverse_IDMAP[prediction['category_id']]
            pred_annotations.append(prediction)



     
        # Convert _predictions to JSON format
        predictions_json = json.dumps(pred_annotations)
        
            # Write predictions to a JSON file in the current directory
        file_path = "car_sticker_predictions.json"
        with open(file_path, "w") as f:
            f.write(predictions_json)

         #load ground truth
        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            coco_gt = COCO(json_file)

        _evaluate_predictions_on_coco(coco_gt, pred_annotations, 'bbox')

        #comment for now since calculating of AP is done on a separate script 
        #gt_dataset = annotations_to_coco(gt_annotations)



        gt_dataset = None 
        
        

        return gt_dataset, pred_annotations




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



