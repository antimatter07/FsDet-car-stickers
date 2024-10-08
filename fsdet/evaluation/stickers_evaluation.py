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
                    print('processed instance: ',prediction_item)
                    self._predictions.append(prediction_item)
            print("Processed prediction:", prediction)

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

        num_classes = 1  # Assuming there's only one class (car sticker)

        # Initialize variables for true positives, false positives, and false negatives
        tp = np.zeros(num_classes)
        fp = np.zeros(num_classes)
        fn = np.zeros(num_classes)
    
        # Matching predictions with ground truth
        for image_id, pred_annotations_image in gt_dataset.items():
            gt_annotations = gt_dataset.get(image_id, [])
            
            for pred in pred_annotations:
                if pred['image_id'] == image_id:
                    pred_bbox = pred['bbox']
                    pred_class = pred['category_id']
                    pred_score = pred['score']
                    
                    best_iou = 0
                    best_gt_idx = -1
                    
                    for gt_idx, gt in enumerate(gt_annotations):
                        if gt['category_id'] == pred_class:
                            iou = self._calculate_iou(gt['bbox'], pred_bbox)
                            if iou > best_iou:
                                best_iou = iou
                                best_gt_idx = gt_idx
    
                    if best_iou >= 0.5:
                        if 'matched' in gt_annotations[best_gt_idx]:
                            if not gt_annotations[best_gt_idx]['matched']:
                                tp[0] += 1
                                gt_annotations[best_gt_idx]['matched'] = True
                        else:
                            gt_annotations[best_gt_idx]['matched'] = True
                            tp[0] += 1
                    else:
                        fp[0] += 1
    
            # Counting false negatives
            for gt in gt_annotations:
                if 'matched' in gt and not gt['matched']:
                    fn[0] += 1
    
        # Computing precision, recall, and AP
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        ap = self._calculate_ap(precision, recall)
    
        # Displaying results
        ap_result = {
            "AP": ap * 100.0,
        }
        print("Car sticker evaluation results:")
        print("Average Precision (AP): {:.2f}".format(ap_result["AP"]))
    
        self._results.update(ap_result)
        self._logger.info("Car sticker evaluation results: {}".format(self._results))

    def _calculate_iou(self, bbox1, bbox2):
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Calculate intersection coordinates
        x_intersection = max(x1, x2)
        y_intersection = max(y1, y2)
        x_intersection_end = min(x1 + w1, x2 + w2)
        y_intersection_end = min(y1 + h1, y2 + h2)

        # Calculate intersection area
        intersection_area = max(0, x_intersection_end - x_intersection) * max(0, y_intersection_end - y_intersection)

        # Calculate union area
        union_area = w1 * h1 + w2 * h2 - intersection_area

        # Calculate IoU
        iou = intersection_area / union_area

        print("calculated iou: ", iou)

        return iou

    def _calculate_ap(self, precision, recall):
        # Calculate AP using 11-point interpolation
        recall_levels = np.linspace(0, 1, 11)
        interpolated_precision = np.zeros_like(recall_levels)

        for i, rl in enumerate(recall_levels):
            precision_at_recall = precision[recall >= rl]
            if precision_at_recall.size > 0:
                interpolated_precision[i] = np.max(precision_at_recall)
            else:
                interpolated_precision[i] = 0

        ap = np.mean(interpolated_precision)

        return ap

    def _prepare_annotations(self):
        gt_annotations = []
        pred_annotations = []
        processed_image_ids = set()  # Maintain a set to store processed image IDs

        # reverse ID map for 60 base classes + 1 novel (car-sticker), car-sticker is category-id = 91
        metadata = self._metadata
        print('DATASET_NAME:', self._dataset_name)
        is_tiny_only = "tinyonly" in self._dataset_name
        
      
        
      #  if is_tiny_only:
      #      IDMAP = metadata.tinyonly_stickers_id_to_contiguous_id
       # else:
       #     IDMAP = metadata.novel_dataset_id_to_contiguous_id

        #FOR NOW HANDLE CORRECT MAPPINGS FOR EACH DATASET BY MANUALLY CHANGING LINE BELOW
        #  "tinyonly_top4_stickers_id_to_contiguous_id"
        
        #IDMAP = metadata.tinyonly_stickers_id_to_contiguous_id
        IDMAP = metadata.tinyonly_top4_stickers_id_to_contiguous_id
        
        inverse_IDMAP = {v: k for k, v in IDMAP.items()}

        print('**** INVERSE MAP TO BE USED IN EVALUATION *****')
        print(inverse_IDMAP)
        

        for prediction in self._predictions:

            if prediction["image_id"] not in processed_image_ids:
                # Add ground truth annotations only if the image ID hasn't been processed yet
                gt_annotations.extend(get_ground_truth_annotations(prediction["image_id"]))
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
     
        gt_dataset = annotations_to_coco(gt_annotations)



    
        
        

        return gt_dataset, pred_annotations

def get_ground_truth_annotations(image_id):
    annotations = []
    #few-shot-object-detection-0.1/datasets/stickers/annotations/stickers_28shot_test.json
    # datasets/stickers_split/full_box_28shot_train.json
    with open("datasets/stickers/annotations/stickers_28shot_test.json", "r") as json_file:
        your_dataset_json = json.load(json_file)
    for annotation in your_dataset_json["annotations"]:
        if annotation["image_id"] == image_id:
            x_min = annotation["bbox"][0]
            y_min = annotation["bbox"][1]
            width = annotation["bbox"][2]
            height = annotation["bbox"][3]
            #x_max = x_min + width
           # y_max = y_min + height

            annotations.append({
                "image_id": annotation["image_id"],
                "category_id": annotation["category_id"],
                "bbox": [x_min, y_min, width, height],
                "area": width * height,
                "iscrowd": annotation["iscrowd"],
                "segmentation": annotation["segmentation"],
                "id": annotation["id"],
            })
    return annotations

def annotations_to_coco(annotations):
    coco_format_annotations = []
    for annotation in annotations:
        coco_format_annotation = {
            "image_id": annotation["image_id"],
            "category_id": annotation["category_id"],
            "bbox": annotation["bbox"],
            "id": annotation["id"],
            "area": annotation["area"],
            #"segmentation": annotation["segmentation"],
            "iscrowd": annotation["iscrowd"],
        }
        print('anno id being appended: ', coco_format_annotation['id'])
        coco_format_annotations.append(coco_format_annotation)
        
    with open("datasets/stickers/annotations/stickers_28shot_test.json", "r") as json_file:
        your_dataset_json = json.load(json_file)


    gt_dataset = {
        "info": {},
        "licenses": [],
        "images": your_dataset_json["images"],
        "annotations": coco_format_annotations,
        "categories": [{"id": 1, "name": "car-sticker", "supercategory": "car-stickers"}],  
    }


 

    return gt_dataset

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



