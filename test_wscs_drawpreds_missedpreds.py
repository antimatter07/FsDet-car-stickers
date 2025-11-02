"""
test_wscs_drawpreds.py

Implementation of the full windshield-guided detection pipeline.

This script implements the windshield-guided detection pipeline. Given a test dataset and path to weights of 
both the windshield model and sticker model, the script goes through each image and performs inference using
the proposed detection pipeline. 
"""

# --- ACRONYMS ---
# ws = windshield
# cs = car stickers
import json
import os
import cv2
import numpy as np
import torch
from PIL import Image
from fsdet.config import get_cfg
from fsdet.modeling import GeneralizedRCNN
from fsdet.checkpoint import DetectionCheckpointer
from fsdet.data.meta_stickers import register_meta_stickers
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import Boxes, Instances
from detectron2.structures import pairwise_iou, BoxMode
import shutil

from detectron2.utils.visualizer import Visualizer
import detectron2.data.transforms as T

import fsdet.data.builtin # registers all datasets


# Test the FULL IMAGES
input_folder = "datasets/stickers/stickers_ws_test_31shot_1280/" # test image folder
output_folder = "results/with_missed_detections/test_wscs_31shot_images_with_predictions/" # output to save processed images
dataset_name = "stickers_ws_31shot_1280_test_tinyonly_top4" # registered name of the test dataset

#confidence score threshold for each classs
WS_SCORE_THRESHOLD = 0.7
STICKERS_SCORE_THRESHOLD = 0.10

os.makedirs(output_folder, exist_ok=True)
torch.cuda.empty_cache()

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # SET GPU HERE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CURRENT DEVICE: ", device)

# Load FsDet model
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
    
    
    torch.cuda.empty_cache()

    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = weights_path

    model = GeneralizedRCNN(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.to(device).eval()

    return model, cfg


# best ws+stickers config : stickers_ws_31shot_tinyonly_top4_8_random_all_3000iters_lr001_unfreeze_r-nms_fbackbone.yaml
# Weights for best 31shot: results/ws_then_cs_31shot/model_0000299.pth
# Weights for best 2shot: results/ws_then_cs_2shot/model_0001799.pth
# Weights for best 5shot: results/ws_then_cs_5shot/model_0001399.pth
# weights for best 10shot: results/ws_then_cs_10shot/model_0001799.pth

ws_model, ws_cfg = load_model(
    "configs/stickers-detection/stickers_ws_31shot_tinyonly_top4_8_random_all_3000iters_lr001_unfreeze_r-nms_fbackbone.yaml",
    "checkpoints/stickers/stickers_ws_31shot_tinyonly_top4_8_random_all_3000iters_lr001_unfreeze_r-nms_fbackbone/model_final.pth"
)

cs_model, cs_cfg = load_model(
    "configs/stickers-detection/ws_then_cs_31shot.yaml",
    "results/ws_then_cs_31shot/model_0000299.pth"
)


# converts OpenCV image to tensor for GeneralizedRCNN input format
def preprocess_image(image_path):
    image_bgr = cv2.imread(image_path)
    
    if image_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_tensor = torch.from_numpy(image.astype("float32")).permute(2, 0, 1).to(device)

    input = {
        "image": image_tensor,
        "height": image_tensor.shape[1],
        "width": image_tensor.shape[2],
    }

    return input


# Filter predictions to only include desired class
def detect_only_class(img, model, class_id, score_threshold):
    with torch.no_grad():
        outputs = model([img])
        instances = outputs[0]["instances"].to(device)

    # only consider those score above the threshold
    mask = (instances.pred_classes == class_id) & (instances.scores >= score_threshold) 
    filtered_instances = instances[mask]

    return filtered_instances


def create_empty_instances(height, width):
    empty_instances = Instances((height, width))
    empty_instances.pred_boxes = Boxes(torch.empty((0, 4)).to(device))
    empty_instances.scores = torch.empty((0,)).to(device)
    empty_instances.pred_classes = torch.empty((0,), dtype=torch.int64).to(device)

    return empty_instances


def prepare_input_for_model(image_bgr, cfg, device):
    """
    Matches Detectron2's DefaultPredictor preprocessing:
    1. Resize shortest edge.
    2. Convert to float32 tensor.
    3. Pass original height/width for proper scaling.
    """
    transform_gen = T.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST],
        cfg.INPUT.MAX_SIZE_TEST,
    )
    height, width = image_bgr.shape[:2]
    image_resized = transform_gen.get_transform(image_bgr).apply_image(image_bgr)

    # Convert to torch tensor (C, H, W)
    image_tensor = torch.as_tensor(image_resized.astype("float32").transpose(2, 0, 1))

    # Return input dict (model handles normalization internally)
    return {"image": image_tensor.to(device), "height": height, "width": width}
    
@torch.no_grad()
def detect_ws_then_cs(image_bgr, ws_model, cs_model, ws_cfg, cs_cfg, metadata, device="cuda"):
    """
    1. Run windshield model on the full image.
    2. For each windshield detection, crop that region.
    3. Run the sticker model on the crop.
    4. Map sticker detections back to full image coordinates.
    """
    # --- STEP 1: Detect windshields on full image ---
    ws_inputs = prepare_input_for_model(image_bgr, ws_cfg, device)
    ws_outputs = ws_model([ws_inputs])[0]

    ws_instances = detector_postprocess(
        ws_outputs["instances"].to("cpu"), ws_inputs["height"], ws_inputs["width"]
    )
    # Filter windshields by score threshold
    keep = (ws_instances.pred_classes == 5) & (ws_instances.scores >= WS_SCORE_THRESHOLD)
    ws_instances = ws_instances[keep]

    cs_instances_all = []

    # --- STEP 2: For each windshield box, crop and detect stickers ---
    for i, box in enumerate(ws_instances.pred_boxes):
        x1, y1, x2, y2 = map(int, box.tolist())
        crop = image_bgr[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        cs_inputs = prepare_input_for_model(crop, cs_cfg, device)
        cs_outputs = cs_model([cs_inputs])[0]

        cs_instances = detector_postprocess(
            cs_outputs["instances"].to("cpu"), cs_inputs["height"], cs_inputs["width"]
        )

        keep = (cs_instances.pred_classes == 4) & (cs_instances.scores >= STICKERS_SCORE_THRESHOLD)
        cs_instances = cs_instances[keep]

        # --- STEP 3: Shift sticker boxes to full image coordinates ---
        if len(cs_instances) > 0:
            cs_boxes = cs_instances.pred_boxes.tensor
            cs_boxes[:, 0::2] += x1  # shift x1, x2
            cs_boxes[:, 1::2] += y1  # shift y1, y2

            shifted_instances = Instances(
                image_size=image_bgr.shape[:2],
                pred_boxes=Boxes(cs_boxes),
                scores=cs_instances.scores,
                pred_classes=cs_instances.pred_classes,
            )
            cs_instances_all.append(shifted_instances)

    # --- STEP 4: Combine all sticker detections ---
    if len(cs_instances_all) > 0:
        cs_instances = Instances.cat(cs_instances_all)
    else:
        # Create empty instances with all fields defined
        height, width = image_bgr.shape[:2]
        cs_instances = create_empty_instances(height, width)

    return ws_instances, cs_instances

def clean_instance_fields(instances):
    """
    Remove unnecessary tensor labels (pred_masks, etc.) to make visualization cleaner.
    """
    # Clone
    new_instances = Instances(instances.image_size)
    for k, v in instances.get_fields().items():
        new_instances.set(k, v.clone() if torch.is_tensor(v) else v)

    # Remove mask/tensor fields that clutter visualization
    for field in ["pred_masks", "gt_boxes", "gt_classes"]:
        if hasattr(new_instances, field):
            delattr(new_instances, field)

    return new_instances
    

def visualize_result(image_path, ws_instances, cs_instances, draw_missed_gts=False, iou_thresh=0.5):
    """
    Visualizes predictions.
    - Correct detections (IoU ≥ threshold) → green
    - False positives → orange
    - Missed GTs → red
    - Removes random Detectron colors and class text.
    - Uniform white text for labels and confidences.
    """
    image_bgr = cv2.imread(image_path)
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    metadata = MetadataCatalog.get(dataset_name)
    metadata.thing_classes = ["class0", "class1", "class2", "class3", "sticker", "windshield"]

    ws_instances_cpu = clean_instance_fields(ws_instances.to("cpu"))
    cs_instances_cpu = clean_instance_fields(cs_instances.to("cpu"))

    vis = Visualizer(image.copy(), metadata=metadata, scale=1.0)
    vis._default_font_size = 9

    # ------------------------------------------------------------------
    # Custom: draw predictions manually (uniform white color)
    # ------------------------------------------------------------------
    for inst in [ws_instances_cpu, cs_instances_cpu]:
        boxes = inst.pred_boxes.tensor.numpy()
        scores = inst.scores.numpy()
        for box, score in zip(boxes, scores):
            vis.draw_box(box, edge_color=(1.0, 1.0, 1.0))  # white outline
            vis.draw_text(f"{score:.2f}", (box[0], box[1] - 5),
                          color=(1.0, 1.0, 1.0), font_size=6)

    # ------------------------------------------------------------------
    # Evaluate stickers (TP / FP / MISS)
    # ------------------------------------------------------------------
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

            # Determine correct/FP/missed
            max_ious_pred, _ = ious.max(dim=0) if ious.numel() > 0 else (torch.zeros(len(pred_boxes)), None)
            max_ious_gt, _ = ious.max(dim=1) if ious.numel() > 0 else (torch.zeros(len(gt_boxes)), None)

            correct_pred_idxs = (max_ious_pred >= iou_thresh).nonzero(as_tuple=True)[0]
            false_pred_idxs = (max_ious_pred < iou_thresh).nonzero(as_tuple=True)[0]
            missed_gt_idxs = (max_ious_gt < iou_thresh).nonzero(as_tuple=True)[0]

            # --- True Positives: Green
            for idx in correct_pred_idxs:
                box = pred_boxes[idx].numpy()
                conf = cs_instances_cpu.scores[idx].item() if idx < len(cs_instances_cpu.scores) else 0
                vis.draw_box(box, edge_color=(0.0, 1.0, 0.0))
                vis.draw_text(f"TP {conf:.2f}", (box[0], box[1] - 5),
                              color=(1.0, 1.0, 1.0), font_size=6)

            # --- False Positives: Orange
            for idx in false_pred_idxs:
                box = pred_boxes[idx].numpy()
                conf = cs_instances_cpu.scores[idx].item() if idx < len(cs_instances_cpu.scores) else 0
                vis.draw_box(box, edge_color=(1.0, 0.65, 0.0))
                vis.draw_text(f"FP {conf:.2f}", (box[0], box[1] - 5),
                              color=(1.0, 1.0, 1.0), font_size=6)

            # --- Missed GTs: Red
            for idx in missed_gt_idxs:
                box = gt_boxes[idx].numpy()
                vis.draw_box(box, edge_color=(1.0, 0.0, 0.0))
                vis.draw_text("MISS", (box[0], box[1] - 5),
                              color=(1.0, 1.0, 1.0), font_size=6)
        else:
            print(f"[!] No GT sticker boxes found for {image_path}")
    else:
        print(f"[!] No GT found for {image_path}")

    # Save visualization
    output_image = vis.output.get_image()
    output_path = os.path.join(output_folder, os.path.basename(image_path) + "_output.jpg")
    cv2.imwrite(output_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
    print(f"Saved image output to {output_path}")




# Delete all files and subfolders and the ouput folder
def clear_output_folder():
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

# Get all image files from the folder
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith((".jpg", ".jpeg"))]
print("> Image file count: ", len(image_files))
print("> Dataset name: ", dataset_name)

for image_filename in image_files:
    image_path = os.path.join(input_folder, image_filename)
    #image_tensor = preprocess_image(image_path)
    image_bgr = cv2.imread(image_path)
    ws_instances, cs_instances = detect_ws_then_cs(
    image_bgr, ws_model, cs_model, ws_cfg, cs_cfg, MetadataCatalog.get(dataset_name), device=device
    )
    visualize_result(image_path, ws_instances, cs_instances, draw_missed_gts=True)


    #ws_instances, cs_instances = detect_ws_then_cs(image_tensor, ws_model, cs_model, MetadataCatalog.get(dataset_name))

    #visualize_result(image_path, ws_instances, cs_instances) 

