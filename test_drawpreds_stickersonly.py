"""
test_drawpreds_stickersonly.py

Visualize sticker detections only (no windshield model).
Works with Detectron2 v0.4 / FsDet models.
All bounding boxes have uniform white labels.
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

import fsdet.data.builtin # registers all datasets


output_folder = "results/test_10shot_drawpreds_stickersonly/"
dataset_name = "stickers_10shot_1280_test_tinyonly_top4"
input_folder = MetadataCatalog.get(dataset_name).image_root
STICKERS_SCORE_THRESHOLD = 0.00
IOU_THRESH = 0.5

os.makedirs(output_folder, exist_ok=True)
torch.cuda.empty_cache()
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CURRENT DEVICE:", device)



def load_model(config_path, weights_path):
    """
    Loads a Detectron2 model and its configuration for performing inference.

    This function reads a YAML configuration file and initializes a Detectron2
    GeneralizedRCNN model using the specified weights. It also prepares the
    cfg configuration object that defines model and preprocessing parameters.

    Args:
        config_path (str): Path to the YAML file containing the model configuration.
        weights_path (str): Path to the model weights (.pth) file.

    Returns:
        tuple:
            model (torch.nn.Module): The loaded Detectron2 model, set to evaluation mode.
            cfg (CfgNode): The Detectron2 configuration object associated with the model.

    Raises:
        FileNotFoundError: If the specified configuration or weight file does not exist.
        RuntimeError: If model loading or checkpoint restoration fails.
    """
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = weights_path

    model = GeneralizedRCNN(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.to(device).eval()
    return model, cfg


# For 31-shot
# configs/stickers-detection/stickers_only_31shot.yaml
# results/stickers_only/31shot/best/model_0000999.pth

# For 10-shot
# results/stickers_only/10shot/best/model_0003199.pth
#  configs/stickers-detection/stickers_only_10shot.yaml
# For 5-shot
# results/stickers_only/5shot_bs2/best (iter 1400 highest ap50)/model_0001399.pth
# configs/stickers-detection/stickers_only_5shot.yaml
# For 2-shot
# configs/stickers-detection/stickers_only_2shot.yaml
#  results/stickers_only/2shot/best/model_0001799.pth
cs_model, cs_cfg = load_model(
    "configs/stickers-detection/stickers_only_10shot.yaml",
    "results/stickers_only/10shot/best/model_0003199.pth"
)



def prepare_input_for_model(image_bgr, cfg, device):
    """
    Prepare an input image for model inference by resizing and converting to tensor.

    Args:
        image_bgr (numpy.ndarray): Input image in BGR format (H x W x C).
        cfg (CfgNode): Detectron2 configuration object containing preprocessing settings.
        device (torch.device): Device on which the tensor should be located (CPU/GPU).

    Returns:
        dict: A dictionary containing:
            - "image" (torch.Tensor): The preprocessed image tensor (C x H x W) on the specified device.
            - "height" (int): Original image height.
            - "width" (int): Original image width.
    """
    transform_gen = T.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST],
        cfg.INPUT.MAX_SIZE_TEST,
    )
    height, width = image_bgr.shape[:2]
    image_resized = transform_gen.get_transform(image_bgr).apply_image(image_bgr)
    image_tensor = torch.as_tensor(image_resized.astype("float32").transpose(2, 0, 1))
    return {"image": image_tensor.to(device), "height": height, "width": width}



@torch.no_grad()
def detect_stickers(image_bgr, cs_model, cs_cfg):
    """
    Perform sticker detection on an input image using the provided model.

    Args:
        image_bgr (numpy.ndarray): Input image in BGR format (H x W x C).
        cs_model (torch.nn.Module): Sticker detection model in evaluation mode.
        cs_cfg (CfgNode): Configuration object for the sticker detection model.

    Returns:
        Instances: Detectron2 Instances object containing only detected sticker boxes
                   that pass the class and score thresholds.
    """
    
    inputs = prepare_input_for_model(image_bgr, cs_cfg, device)
    outputs = cs_model([inputs])[0]
    instances = outputs["instances"].to("cpu")

    keep = (instances.pred_classes == 4) & (instances.scores >= STICKERS_SCORE_THRESHOLD)
    instances = instances[keep]
    return instances



def clean_instance_fields(instances):
    """
    Create a copy of an Instances object with unnecessary fields removed.

    Specifically removes 'pred_masks', 'gt_boxes', and 'gt_classes' if present.

    Args:
        instances (Instances): Original Detectron2 Instances object.

    Returns:
        Instances: Cleaned Instances object containing only relevant fields.
    """
    
    new_instances = Instances(instances.image_size)
    for k, v in instances.get_fields().items():
        new_instances.set(k, v.clone() if torch.is_tensor(v) else v)
    for field in ["pred_masks", "gt_boxes", "gt_classes"]:
        if hasattr(new_instances, field):
            delattr(new_instances, field)
    return new_instances


def visualize_result(image_path, cs_instances, draw_missed_gts=True):
    """
    Visualize detected stickers on an image, saving output with true positives, false positives, and missed ground truths.

    Draws:
        - Predicted boxes in white by default.
        - True Positives (TP) in green.
        - False Positives (FP) in orange.
        - Missed ground truths in red.

    Args:
        image_path (str): Path to the input image.
        cs_instances (Instances): Sticker detection results for the image.
        draw_missed_gts (bool, optional): Whether to mark missed ground truth boxes. Defaults to True.

    Saves:
        Output image to `output_folder` with "_output.jpg" appended to the original filename.
    """
    image_bgr = cv2.imread(image_path)
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    metadata = MetadataCatalog.get(dataset_name)
    metadata.thing_classes = ["class0", "class1", "class2", "class3", "sticker", "windshield"]

    cs_instances_cpu = clean_instance_fields(cs_instances.to("cpu"))
    vis = Visualizer(image.copy(), metadata=metadata, scale=1.0)
    vis._default_font_size = 9

    # Draw all predicted boxes in white
    for box, score in zip(cs_instances_cpu.pred_boxes.tensor.numpy(), cs_instances_cpu.scores.numpy()):
        vis.draw_box(box, edge_color=(1.0, 1.0, 1.0))
        vis.draw_text(f"{score:.2f}", (box[0], box[1] - 5), color=(1.0, 1.0, 1.0), font_size=6)

    # Load ground truths to mark TP/FP/Missed
    dataset_dicts = DatasetCatalog.get(dataset_name)
    img_gt = next((d for d in dataset_dicts
                   if os.path.basename(d["file_name"]) == os.path.basename(image_path)), None)

    if img_gt is not None and "annotations" in img_gt:
        gt_boxes = []
        for ann in img_gt["annotations"]:
            if ann.get("category_id", -1) == 4:  # sticker class
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

            # True Positives: Green
            for idx in correct_pred_idxs:
                box = pred_boxes[idx].numpy()
                conf = cs_instances_cpu.scores[idx].item()
                vis.draw_box(box, edge_color=(0.0, 1.0, 0.0))
                vis.draw_text(f"{conf:.2f}", (box[0], box[1] - 5), color=(1.0, 1.0, 1.0), font_size=6)

            # False Positives: Orange
            for idx in false_pred_idxs:
                box = pred_boxes[idx].numpy()
                conf = cs_instances_cpu.scores[idx].item()
                vis.draw_box(box, edge_color=(1.0, 0.65, 0.0))
                vis.draw_text(f"{conf:.2f}", (box[0], box[1] - 5), color=(1.0, 1.0, 1.0), font_size=6)

            # Missed GTs: Red
            for idx in missed_gt_idxs:
                box = gt_boxes[idx].numpy()
                vis.draw_box(box, edge_color=(1.0, 0.0, 0.0))
                vis.draw_text("MISS", (box[0], box[1] - 5), color=(1.0, 1.0, 1.0), font_size=6)

    output_image = vis.output.get_image()
    output_path = os.path.join(output_folder, os.path.basename(image_path) + "_output.jpg")
    cv2.imwrite(output_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
    print(f"Saved: {output_path}")



def clear_output_folder():
    """
    Delete all files and subdirectories in the output folder to ensure a clean start.

    If the folder does not exist, it will be created.
    """
    if os.path.exists(output_folder):
        for filename in os.listdir(output_folder):
            file_path = os.path.join(output_folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        os.makedirs(output_folder)


clear_output_folder()
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith((".jpg", ".jpeg"))]
print("> Image file count:", len(image_files))
print("> Dataset name:", dataset_name)

for image_filename in image_files:
    image_path = os.path.join(input_folder, image_filename)
    image_bgr = cv2.imread(image_path)
    cs_instances = detect_stickers(image_bgr, cs_model, cs_cfg)
    visualize_result(image_path, cs_instances, draw_missed_gts=True)
