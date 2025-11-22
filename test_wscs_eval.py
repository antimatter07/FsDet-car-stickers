import os
import sys
import json
from collections import defaultdict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


cs_category_id = 91
ws_category_id = 92
    
def compute_iou(box1, box2):
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.
    
    Args:
        box1 (list[float]): [x, y, width, height] of the first box.
        box2 (list[float]): [x, y, width, height] of the second box.
        
    Returns:
        float: IoU value between 0 and 1. Returns 0 if union is zero.
    """
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



def get_alignment(sticker_box, ws_boxes):
    """
    Determine the alignment ('left' or 'right') of a sticker relative to windshields.
    
    Args:
        sticker_box (list[float]): Bounding box of the sticker [x, y, width, height].
        ws_boxes (list[list[float]]): List of bounding boxes of windshields in the image.
    
    Returns:
        str: "left", "right", or "unknown" if no windshield is found.
    """
    best_iou = 0
    best_ws = None

    for ws in ws_boxes:
        iou = compute_iou(sticker_box, ws)
        if iou > best_iou:
            best_iou = iou
            best_ws = ws

    if best_ws is None:
        return "unknown"

    # compute relative to chosen windshield
    wx1, wy1, wx2, wy2 = best_ws
    ws_center = wx1 + wx2 / 2

    sx1, sy1, sx2, sy2 = sticker_box
    s_center = sx1 + sx2 / 2

    return "left" if s_center < ws_center else "right"


# create a lookup of ground truths per image
def load_gt_by_image(category_id):
    """
    Load ground truth boxes and organize them by image ID.
    
    Args:
        category_id (int): COCO category ID to filter annotations.
    
    Returns:
        dict: {image_id: [list of bounding boxes]}
    """
    with open(gt_json) as f:
        gt = json.load(f)

    image_gt = defaultdict(list)

    for ann in gt["annotations"]:
        if ann["category_id"] == category_id:
            image_gt[ann["image_id"]].append(ann["bbox"])

    return dict(image_gt)


# create a lookup of predictions per image
def load_pred_by_image(category_id=cs_category_id):
    """
    Load predicted boxes and organize them by image ID, sorted by confidence.
    
    Args:
        category_id (int, optional): COCO category ID to filter predictions. Defaults to cs_category_id.
    
    Returns:
        dict: {image_id: [(bbox, score), ...]} sorted descending by score.
    """
    with open(pred_json) as f:
        predictions = json.load(f)

    image_pred = defaultdict(list)
    
    for p in predictions:
        if p["category_id"] == category_id:
            image_pred[p["image_id"]].append((p["bbox"], p["score"]))

    # sort by confidence desc
    for preds in image_pred.values():
        preds.sort(key=lambda x: x[1], reverse=True)

    return dict(image_pred)
    

# compare gt and prediction boxes in an image (greedy method)
def compare_boxes(gt_boxes, pred_boxes, ws_boxes, iou_thresh=0.5):
    """
    Compare predicted boxes with ground truth using a greedy IoU matching.
    
    Args:
        gt_boxes (list[list[float]]): List of ground truth bounding boxes.
        pred_boxes (list[tuple]): List of predicted bounding boxes with scores [(bbox, score)].
        ws_boxes (list[list[float]]): List of windshield bounding boxes.
        iou_thresh (float, optional): IoU threshold to count as true positive. Defaults to 0.5.
    
    Returns:
        tuple: tp, fp, fn, tp_left, tp_right, fp_left, fp_right, fn_left, fn_right
    """
    matched_gt = set()
    tp = 0
    fp = 0
    tp_left = tp_right = 0
    fn_left = fn_right = 0
    fp_left = fp_right = 0

    # determine car sticker GT alignments
    gt_alignments = [get_alignment(g, ws_boxes) for g in gt_boxes]
    
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
            if gt_alignments[best_gi] == "left":
                tp_left += 1
            else:
                tp_right += 1
        else:
            fp+=1
            fp_alignment = get_alignment(p_box, ws_boxes)
            if fp_alignment == "left":
                fp_left += 1
            else:
                fp_right += 1

    fn = len(gt_boxes) - len(matched_gt)

    # count false negatives per side
    for gi, g_side in enumerate(gt_alignments):
        if gi not in matched_gt:
            if g_side == "left":
                fn_left += 1
            else:
                fn_right += 1

    return tp, fp, fn, tp_left, tp_right, fp_left, fp_right, fn_left, fn_right
                

# Evaluate all predictions
def evaluate_images(img_ids, iou_thresh=0.5):
    """
    Evaluate predictions for a list of images and compute precision/recall metrics.
    
    Args:
        img_ids (list[int]): List of COCO image IDs to evaluate.
        iou_thresh (float, optional): IoU threshold for true positives. Defaults to 0.5.
    
    Returns:
        dict: Contains mean precision, mean recall, and per-side precision/recall.
    """
    
    ws_gt = load_gt_by_image(ws_category_id)
    cs_gt = load_gt_by_image(cs_category_id)
    pred = load_pred_by_image()

    precisions = []
    recalls = []

    total_gt_left = total_gt_right = 0
    total_tp_left = total_tp_right = 0
    total_fp_left = total_fp_right = 0

    for image_id in img_ids:
        g = cs_gt.get(image_id, [])
        p = pred.get(image_id, [])
        ws_boxes = ws_gt[image_id]
        
        tp, fp, fn, tp_left, tp_right, fp_left, fp_right, fn_left, fn_right = compare_boxes(g, p, ws_boxes, iou_thresh)
        prec = tp / (tp + fp + 1e-6)
        rec  = tp / (tp + fn + 1e-6)
        precisions.append(prec)
        recalls.append(rec)
        
        total_gt_left  += (tp_left + fn_left)
        total_gt_right += (tp_right + fn_right)
        
        total_tp_left  += tp_left
        total_tp_right += tp_right

        total_fp_left  += fp_left
        total_fp_right += fp_right

    mP = sum(precisions) / len(precisions)
    mR = sum(recalls) / len(recalls)

    prec_left  = total_tp_left  / (total_tp_left  + total_fp_left  + 1e-6)
    prec_right = total_tp_right / (total_tp_right + total_fp_right + 1e-6)
    
    rec_left  = total_tp_left  / (total_gt_left  + 1e-6)
    rec_right = total_tp_right / (total_gt_right + 1e-6)

    
    return dict(
        iou_thresh = iou_thresh,
        mean_precision = mP,
        precision_left = prec_left,
        precision_right = prec_right,
        mean_recall = mR,
        recall_left = rec_left,
        recall_right = rec_right
    )


def coco_filter_by_alignment(coco_gt, coco_pred, img_ids, desired_side):
    """
    Filter COCO ground truth and predictions by alignment ('left' or 'right').
    
    Args:
        coco_gt (COCO): COCO object for ground truth.
        coco_pred (COCO): COCO object for predictions.
        img_ids (list[int]): List of image IDs to filter.
        desired_side (str): "left" or "right".
    
    Returns:
        tuple: (filtered_gt, filtered_pred)
            filtered_gt: dict with 'images', 'annotations', 'categories'
            filtered_pred: list of prediction annotations
    """
    filtered_gt = {"images": [], "annotations": [], "categories": coco_gt.dataset["categories"]}
    filtered_pred = []

    ws_gt = load_gt_by_image(ws_category_id)

    idset = set(img_ids)
    filtered_gt["images"] = [img for img in coco_gt.dataset["images"] if img["id"] in idset]

    # filter GT
    for ann in coco_gt.dataset["annotations"]:
        if ann["image_id"] not in idset:
            continue
        if ann["category_id"] != cs_category_id:
            continue

        ws_boxes = ws_gt.get(ann["image_id"], [])
        side = get_alignment(ann["bbox"], ws_boxes)

        if side == desired_side:
            filtered_gt["annotations"].append(ann)

    # filter predictions
    for p in coco_pred.dataset["annotations"]:
        if p["image_id"] not in idset:
            continue
        if p["category_id"] != cs_category_id:
            continue

        ws_boxes = ws_gt.get(p["image_id"], [])
        side = get_alignment(p["bbox"], ws_boxes)

        if side == desired_side:
            filtered_pred.append(p)

    return filtered_gt, filtered_pred





if __name__ == "__main__":
    
    # pred_json= "results/ws_then_cs_31shot/test_predictions/test_predictions.json"
    pred_json = "results/stickers_only/2shot/test_predictions/test_predictions.json"
    gt_json = "datasets/stickers/annotations/stickers_ws_31shot_test_1280.json"
    
    # Output file
    # output_folder = "results/ws_then_cs_31shot/test_predictions/eval/"
    output_folder = "results/stickers_only/2shot/test_predictions/eval/"
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, "view_eval_results.txt")
    sys.stdout = open(output_file, "w")
    
    
    coco_gt = COCO(gt_json)
    coco_pred = coco_gt.loadRes(pred_json)
    
    # Group by view
    view_to_imgids = defaultdict(list)

    
    for img in coco_gt.dataset["images"]:
        tags = img.get("extra", {}).get("user_tags", [])
        view = tags[0] if tags else "_no_tag"
        view_to_imgids[view].append(img["id"])

    print("======================= EVALUATION PER VIEW =======================")

    for view, img_ids in sorted(view_to_imgids.items()):
        print(f">> Evaluating View: {view} ({len(img_ids)} images)")
    
        res = evaluate_images(img_ids)
    
        for key, value in res.items():
            print(f"{key}: {value:.3f}")
    
        print("COCO EVAL COMBINED: ")
        coco_eval = COCOeval(coco_gt, coco_pred, iouType="bbox")
        coco_eval.params.imgIds = img_ids
        coco_eval.params.catIds = [cs_category_id]
        coco_eval.evaluate(); coco_eval.accumulate(); coco_eval.summarize()
    
        # print("COCO EVAL LEFT: ")
        # gt_left, pred_left = coco_filter_by_alignment(coco_gt, coco_pred, img_ids, "left")
        # coco_gt_left = COCO(); coco_gt_left.dataset = gt_left; coco_gt_left.createIndex()
        # coco_pred_left = coco_gt_left.loadRes(pred_left)
        # ev_left = COCOeval(coco_gt_left, coco_pred_left, "bbox")
        # ev_left.evaluate(); ev_left.accumulate(); ev_left.summarize()
    
    
        # print("COCO EVAL RIGHT: ")
        # gt_right, pred_right = coco_filter_by_alignment(coco_gt, coco_pred, img_ids, "right")
        # coco_gt_right = COCO(); coco_gt_right.dataset = gt_right; coco_gt_right.createIndex()
        # coco_pred_right = coco_gt_right.loadRes(pred_right)
        # ev_right = COCOeval(coco_gt_right, coco_pred_right, "bbox")
        # ev_right.evaluate(); ev_right.accumulate(); ev_right.summarize()
        
        print("\n========================================\n")

