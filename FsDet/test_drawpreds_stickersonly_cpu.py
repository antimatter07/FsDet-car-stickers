"""
test_drawpreds_stickersonly_cpu.py

CPU-only visualization script for FsDet sticker detections.
"""

import os
import cv2
import torch
import shutil
import numpy as np
from fsdet.config import get_cfg
from fsdet.modeling import GeneralizedRCNN
from fsdet.checkpoint import DetectionCheckpointer
from detectron2.structures import Boxes, Instances, BoxMode, pairwise_iou
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import detectron2.data.transforms as T

import fsdet.data.builtin  # registers datasets



def load_model(config_path, weights_path):
    """
    Loads a Detectron2 model in CPU mode.

    Returns:
        model (nn.Module) – Model on CPU, eval() mode.
        cfg (CfgNode) – Loaded config.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.DEVICE = "cpu"   # 🔥 FORCE CPU MODE

    model = GeneralizedRCNN(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()               # no .to(device), already CPU

    return model, cfg



def prepare_input_for_model(image_bgr, cfg):
    """Prepare input image for Detectron2 model (CPU)."""

    transform_gen = T.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST],
        cfg.INPUT.MAX_SIZE_TEST,
    )

    height, width = image_bgr.shape[:2]
    image_resized = transform_gen.get_transform(image_bgr).apply_image(image_bgr)
    image_tensor = torch.as_tensor(image_resized.astype("float32").transpose(2, 0, 1))

    return {"image": image_tensor, "height": height, "width": width}



@torch.no_grad()
def detect_stickers(image_bgr, cs_model, cs_cfg):
    """Runs sticker detection on CPU."""

    inputs = prepare_input_for_model(image_bgr, cs_cfg)
    outputs = cs_model([inputs])[0]
    instances = outputs["instances"].to("cpu")

    keep = (instances.pred_classes == 4) & (instances.scores >= STICKERS_SCORE_THRESHOLD)
    return instances[keep]



def clean_instance_fields(instances):
    """Remove unused fields from Instances."""
    new_instances = Instances(instances.image_size)

    for k, v in instances.get_fields().items():
        new_instances.set(k, v.clone() if torch.is_tensor(v) else v)

    for field in ["pred_masks", "gt_boxes", "gt_classes"]:
        if hasattr(new_instances, field):
            delattr(new_instances, field)

    return new_instances



def visualize_result(image_path, cs_instances, draw_missed_gts=True):
    """Draw TP/FP/missed GTs on image."""

    image_bgr = cv2.imread(image_path)
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    metadata = MetadataCatalog.get(dataset_name)
    metadata.thing_classes = ["class0", "class1", "class2", "class3", "sticker", "windshield"]

    cs_instances_cpu = clean_instance_fields(cs_instances.to("cpu"))

    vis = Visualizer(image.copy(), metadata=metadata, scale=1.0)
    vis._default_font_size = 9

    # Draw predictions in white
    for box, score in zip(cs_instances_cpu.pred_boxes.tensor.numpy(), cs_instances_cpu.scores.numpy()):
        vis.draw_box(box, edge_color=(1, 1, 1))
        vis.draw_text(f"{score:.2f}", (box[0], box[1] - 5), color=(1, 1, 1), font_size=6)

    # Load ground truths
    dataset_dicts = DatasetCatalog.get(dataset_name)
    img_gt = next((d for d in dataset_dicts
                   if os.path.basename(d["file_name"]) == os.path.basename(image_path)), None)

    if img_gt is not None and "annotations" in img_gt:
        gt_boxes = []
        for ann in img_gt["annotations"]:
            if ann.get("category_id", -1) == 4:
                bbox = ann["bbox"]
                if ann["bbox_mode"] != BoxMode.XYXY_ABS:
                    bbox = BoxMode.convert(bbox, ann["bbox_mode"], BoxMode.XYXY_ABS)
                gt_boxes.append(bbox)

        gt_boxes = torch.tensor(gt_boxes, dtype=torch.float32) if gt_boxes else torch.empty((0, 4))

        if len(gt_boxes) > 0:
            pred_boxes = cs_instances_cpu.pred_boxes.tensor if len(cs_instances_cpu) > 0 else torch.empty((0, 4))
            ious = pairwise_iou(Boxes(gt_boxes), Boxes(pred_boxes)) if len(pred_boxes) > 0 else torch.zeros((len(gt_boxes), 0))

            max_ious_pred, _ = ious.max(dim=0) if ious.numel() > 0 else (torch.zeros(len(pred_boxes)), None)
            max_ious_gt, _ = ious.max(dim=1) if ious.numel() > 0 else (torch.zeros(len(gt_boxes)), None)

            correct_pred_idxs = (max_ious_pred >= IOU_THRESH).nonzero(as_tuple=True)[0]
            false_pred_idxs = (max_ious_pred < IOU_THRESH).nonzero(as_tuple=True)[0]
            missed_gt_idxs = (max_ious_gt < IOU_THRESH).nonzero(as_tuple=True)[0]

            # TP: Green
            for idx in correct_pred_idxs:
                box = pred_boxes[idx].numpy()
                score = cs_instances_cpu.scores[idx].item()
                vis.draw_box(box, edge_color=(0, 1, 0))
                vis.draw_text(f"{score:.2f}", (box[0], box[1] - 5), color=(1, 1, 1), font_size=6)

            # FP: Orange
            for idx in false_pred_idxs:
                box = pred_boxes[idx].numpy()
                score = cs_instances_cpu.scores[idx].item()
                vis.draw_box(box, edge_color=(1, 0.65, 0))
                vis.draw_text(f"{score:.2f}", (box[0], box[1] - 5), color=(1, 1, 1), font_size=6)

            # Missed GT: Red
            for idx in missed_gt_idxs:
                vis.draw_box(gt_boxes[idx].numpy(), edge_color=(1, 0, 0))
                vis.draw_text("MISS", (gt_boxes[idx][0], gt_boxes[idx][1] - 5),
                              color=(1, 1, 1), font_size=6)

    output_img = vis.output.get_image()
    output_path = os.path.join(output_folder, os.path.basename(image_path) + "_output.jpg")
    cv2.imwrite(output_path, cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
    print(f"Saved: {output_path}")



def clear_output_folder():
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)



if __name__ == "__main__":

    output_folder = "results/test_10shot_drawpreds_stickersonly/"
    dataset_name = "stickers_10shot_1280_test_tinyonly_top4"
    input_folder = MetadataCatalog.get(dataset_name).image_root

    STICKERS_SCORE_THRESHOLD = 0.05
    IOU_THRESH = 0.5

    device = torch.device("cpu")
    print("CURRENT DEVICE:", device)

    clear_output_folder()

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith((".jpg", ".jpeg"))]
    print("> Image file count:", len(image_files))
    print("> Dataset name:", dataset_name)

    cs_model, cs_cfg = load_model(
        "configs/stickers-detection/stickers_only_10shot.yaml",
        "results/stickers_only/10shot/best/model_0003199.pth"
    )

    for img in image_files:
        path = os.path.join(input_folder, img)
        img_bgr = cv2.imread(path)
        cs_instances = detect_stickers(img_bgr, cs_model, cs_cfg)
        visualize_result(path, cs_instances)