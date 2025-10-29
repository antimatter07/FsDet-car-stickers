import json
import torch
from collections import defaultdict

pred_annot = "results/"
gt_annot = "datasets/stickers/annotations/stickers_ws_31shot_test_1280.json" # includes GT boxes and view angle of the image

# Load dataset & create lookup tables for comparison of predictions and GT values
def load_dataset(dataset_annot_path):
    with open(dataset_annot_path, "r") as f:
        dataset_annot = json.load(f)
    
    id_info_lookup = {image["id"]: image for image in dataset_annot["images"]}
    name_id_lookup = {image["file_name"]: image["id"] for image in dataset_annot["images"]}
    
    image_gt_dict = defaultdict(list)
    for annot in dataset_annot["annotations"]:
        x1, y1, w, h = annot["bbox"]
        x2, y2 = x1 + w, y1 + h
        image_gt_dict[annot["image_id"]].append([x1, y1, x2, y2])

    return id_info_lookup, name_id_lookup, image_gt_dict


# IoU computation
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)



# Main evaluator
# Evaluates the accuracy of image
def evaluate_image(image_filename, cs_instances, id_info_lookup, name_id_lookup, image_gt_dict):
    image_id = name_id_lookup.get(image_filename)
    if image_id is None:
        print(f"Error: {image_filename} not found in dataset annotation.")
        return None

    img_info = id_info_lookup[image_id]
    extra = img_info.get("extra", {})
    tags = extra.get("tags", [])
    view_angle = tags[0] if tags else "unknown"

    gt_boxes = image_gt_dict[image_id]

    pred_boxes = cs_instances.pred_boxes.tensor.cpu().numpy().tolist()

    ious = []
    for gt in gt_boxes:
        best_iou = 0
        for pred in pred_boxes:
            iou = compute_iou(gt, pred)
            best_iou = max(best_iou, iou)
        ious.append(best_iou)

    avg_iou = sum(ious) / len(ious) if ious else 0.0
    acc = (sum(i >= 0.5 for i in ious) / len(ious) * 100) if ious else 0.0

    print(f"\nImage: {image_filename}")
    print(f"View angle: {view_angle}")
    print(f"Average IoU: {avg_iou:.3f}")
    print(f"Detection accuracy (IoU@0.5): {acc:.2f}%")

    return {
        "image": image_filename,
        "view_angle": view_angle,
        "avg_iou": avg_iou,
        "accuracy": acc,
    }
