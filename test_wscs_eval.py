import json
import torch
import numpy as np
from collections import defaultdict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

pred_json= "results/test_evaluation/test_predictions.json"
gt_json = "datasets/stickers/annotations/stickers_ws_31shot_test_1280.json" # includes view angle of the image


def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # compute intersection
    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)
    
    inter = max(0, xb - xa) * max(0, yb - ya)
    union = (w1 * h1) + (w2 * h2) - inter

    if union == 0:
        return 0.0
    
    return inter / union
    
# create a lookup of ground truths per image
def load_gt_by_image(category_id=91): #car sticker
    with open(gt_json) as f:
        gt = json.load(f)

    image_gt = defaultdict(list)

    for ann in gt["annotations"]:
        if ann["category_id"] == category_id:
            image_gt[ann["image_id"]].append(ann["bbox"])

    return dict(image_gt)


# create a lookup of predictions per image
def load_pred_by_image(category_id=4): #car sticker
    with open(pred_json) as f:
        predictions = json.load(f)

    with open(gt_json) as f:
        gt = json.load(f)

    image_id_lookup = { 
        image["file_name"] : image["id"] for image in gt["images"]
    }

    image_pred = defaultdict(list)
    
    for p in predictions:
        if p["category_id"] == category_id:
            image_id = image_id_lookup[p["file_name"]]
            image_pred[image_id].append((p["bbox"], p["score"]))

    # sort by confidence desc
    for preds in image_pred.values():
        preds.sort(key=lambda x: x[1], reverse=True)

    return dict(image_pred)


# compare gt and prediction boxes in an image (greedy method)
def compare_boxes(gt_boxes, pred_boxes, iou_thresh=0.5):
    matched_gt = set()
    tp = 0
    fp = 0

    for p_box, _ in pred_boxes:
        best_iou = 0
        best_gi = -1
        
        # find the best gt box match for this prediction
        for gi, g_box in enumerate(gt_boxes):
            if gi in matched_gt:
                continue
        
            iou = compute_iou(p_box, g_box)
            if iou > best_iou:
                best_iou = iou
                best_gi = gi
        if best_iou >= iou_thresh:
            tp+=1
            matched_gt.add(best_gi)
        else:
            fp+=1         

    fn = len(gt_boxes) - len(matched_gt)

    return tp, fp, fn
                

# Evaluate all predictions
def evaluate(iou_thresh=0.5): # TODO: add view variable
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    gt = load_gt_by_image()
    predictions = load_pred_by_image()
    
    total_tp = total_fp = total_fn = 0
    
    image_ids = set(gt.keys()) | set(predictions.keys())
    for image_id in image_ids:

        gt_boxes = gt.get(image_id, [])
        pred_boxes = predictions.get(image_id, [])

        tp, fp, fn = compare_boxes(gt_boxes, pred_boxes, iou_thresh)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall    = total_tp / (total_tp + total_fn + 1e-6)

    print("=== Car Sticker Detection Evaluation ===")
    print(f"True Positives : {total_tp}")
    print(f"False Positives: {total_fp}")
    print(f"False Negatives: {total_fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")

    return total_tp, total_fp, total_fn, precision, recall
                    


evaluate()
