import os
import sys
import json
from collections import defaultdict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

pred_json= "results/ws_then_cs_31shot/test_predictions/test_predictions.json"
gt_json = "datasets/stickers/annotations/stickers_ws_31shot_test_1280.json"

cs_category_id = 91
ws_category_id = 92

# Output file
output_folder = "results/ws_then_cs_31shot/test_predictions/eval/"
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


# determines alignment of a GT sticker
# assumes that multiple windshields per image is possible
def get_alignment(sticker_box, ws_boxes):
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
    wx, wy, ww, wh = best_ws
    ws_center = wx + ww/2

    sx, sy, sw, sh = sticker_box
    s_center = sx + sw/2

    return "left" if s_center < ws_center else "right"


# create a lookup of ground truths per image
def load_gt_by_image(category_id):
    with open(gt_json) as f:
        gt = json.load(f)

    image_gt = defaultdict(list)

    for ann in gt["annotations"]:
        if ann["category_id"] == category_id:
            image_gt[ann["image_id"]].append(ann["bbox"])

    return dict(image_gt)


# create a lookup of predictions per image
def load_pred_by_image(category_id=cs_category_id):
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
    matched_gt = set()
    tp = 0
    fp = 0
    tp_left = tp_right = 0
    fn_left = fn_right = 0

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
    ws_gt = load_gt_by_image(ws_category_id)
    cs_gt = load_gt_by_image(cs_category_id)
    pred = load_pred_by_image()

    APs = []
    ARs = []

    total_gt_left = total_gt_right = 0
    total_tp_left = total_tp_right = 0
    AR50_left = AR50_right = 0

    for image_id in img_ids:
        g = cs_gt.get(image_id, [])
        p = pred.get(image_id, [])
        ws_boxes = ws_gt[image_id]
        
        tp, fp, fn, tp_left, tp_right, fp_left, fp_right, fn_left, fn_right = compare_boxes(g, p, ws_boxes, iou_thresh)
        prec = tp / (tp + fp + 1e-6)
        rec  = tp / (tp + fn + 1e-6)
        APs.append(prec)
        ARs.append(rec)
        
        total_gt_left  += (tp_left + fn_left)
        total_gt_right += (tp_right + fn_right)
        
        total_tp_left  += tp_left
        total_tp_right += tp_right

    AP = sum(APs) / len(APs)
    AR = sum(ARs) / len(ARs)

    acc_left  = total_tp_left  / ____
    acc_right = total_tp_right / ___
    
    rec_left  = total_tp_left  / (total_gt_left  + 1e-6)
    rec_right = total_tp_right / (total_gt_right + 1e-6)

    
    return dict(
        iou_thresh = iou_thresh,
        AP = AP,
        AR = AR,
        rec_left = rec_left,
        rec_right = rec_right
    )



print("======================= EVALUATION PER VIEW =======================")

for view, img_ids in sorted(view_to_imgids.items()):
    print(f">> Evaluating View: {view} ({len(img_ids)} images)")

    res = evaluate_images(img_ids)

    for key, value in res.items():
        print(f"{key}: {value:.3f}")

    print("COCO EVAL: ")
    coco_eval = COCOeval(coco_gt, coco_pred, iouType="bbox")
    coco_eval.params.imgIds = img_ids
    coco_eval.params.catIds = [cs_category_id]
    
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    print("\n========================================\n")

