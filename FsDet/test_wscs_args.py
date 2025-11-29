import argparse
import json
import os
import cv2
import torch
from PIL import Image

from fsdet.config import get_cfg
from fsdet.modeling import GeneralizedRCNN
from fsdet.checkpoint import DetectionCheckpointer
from fsdet.data.meta_stickers import register_meta_stickers

from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.structures import Boxes, Instances
import detectron2.data.transforms as T

import fsdet.data.builtin  # registers datasets



def get_args():
    parser = argparse.ArgumentParser(description="Windshield-to-Sticker Detection Pipeline")

    # REQUIRED MODEL CONFIGS
    parser.add_argument("--ws-config", type=str, required=True,
                        help="Path to windshield YAML config")
    parser.add_argument("--ws-weights", type=str, required=True,
                        help="Path to windshield weights (.pth)")
    parser.add_argument("--cs-config", type=str, required=True,
                        help="Path to car-sticker YAML config")
    parser.add_argument("--cs-weights", type=str, required=True,
                        help="Path to car-sticker weights (.pth)")

    # IO
    parser.add_argument("--input-folder", type=str, required=True,
                        help="Folder containing test images")
    parser.add_argument("--output-folder", type=str, required=True,
                        help="Folder to save output JSON")
    parser.add_argument("--gt-json", type=str, required=True,
                        help="Ground-truth COCO JSON")

    parser.add_argument("--dataset-name", type=str,
                        default="stickers_ws_31shot_1280_test_tinyonly_top4")

    # SCORE THRESHOLDS
    parser.add_argument("--ws-score-thresh", type=float, default=0.7)
    parser.add_argument("--cs-score-thresh", type=float, default=0.05)

    # MODE
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["ws-then-cs", "cs-only"],
        help=(
            "Pipeline mode:\n"
            "  ws-then-cs : detect windshield -> detect stickers inside\n"
            "  cs-only    : detect stickers over full image"
        )
    )

    return parser.parse_args()



def load_model(config_path, weights_path, device):
    """
    Load a FsDet/Detectron2 GeneralizedRCNN model with specified config and weights.
    
    Args:
        config_path (str): Path to the config YAML file.
        weights_path (str): Path to the pre-trained weights file (.pth).
    
    Returns:
        tuple: (model, cfg)
            model: Loaded model in evaluation mode on GPU/CPU.
            cfg: Configuration object used for the model.
    """
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = weights_path

    model = GeneralizedRCNN(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

    model.to(device).eval()
    return model, cfg



def create_empty_instances(height, width, device):
    """
    Create an empty Instances object with the required fields for predictions.
    
    Args:
        height (int): Image height.
        width (int): Image width.
    
    Returns:
        Instances: Empty Instances object with pred_boxes, scores, pred_classes initialized.
    """
    empty = Instances((height, width))
    empty.pred_boxes = Boxes(torch.empty((0, 4)).to(device))
    empty.scores = torch.empty((0,), device=device)
    empty.pred_classes = torch.empty((0,), dtype=torch.int64, device=device)
    return empty


def prepare_input_for_model(image_bgr, cfg, device):
    """
    Prepare an input dictionary for a Detectron2/FsDet model.
    
    Steps:
        1. Resize shortest edge based on cfg.
        2. Convert image to float32 torch tensor in (C, H, W) format.
    
    Args:
        image_bgr (numpy.ndarray): Input image in BGR format.
        cfg (CfgNode): FsDet/Detectron2 config object.
        device (str/torch.device): Device to place tensor on.
    
    Returns:
        dict: {'image': tensor, 'height': original height, 'width': original width}
    """
    transform = T.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST],
        cfg.INPUT.MAX_SIZE_TEST,
    )

    height, width = image_bgr.shape[:2]
    resized = transform.get_transform(image_bgr).apply_image(image_bgr)

    tensor = torch.as_tensor(resized.astype("float32").transpose(2, 0, 1))
    return {"image": tensor.to(device), "height": height, "width": width}



@torch.no_grad()
def detect_ws_then_cs(image_bgr, ws_model, cs_model, ws_cfg, cs_cfg,
                      ws_thresh, cs_thresh, device):
    """
    Detect windshields and then car stickers in cropped windshield regions.
    
    Steps:
        1. Detect windshields in full image.
        2. Crop each windshield and detect stickers.
        3. Shift sticker coordinates to full image.
        4. Combine all sticker detections.
    
    Args:
        image_bgr (numpy.ndarray): Input image in BGR format.
        ws_model (nn.Module): Windshield detection model.
        cs_model (nn.Module): Car sticker detection model.
        ws_cfg (CfgNode): Windshield model config.
        cs_cfg (CfgNode): Sticker model config.
        metadata (MetadataCatalog): Dataset metadata.
        device (str): Device to run model on.
    
    Returns:
        tuple: (ws_instances, cs_instances)
            ws_instances: Instances object of detected windshields.
            cs_instances: Instances object of detected stickers with 'alignments'.
    """

    # Windshield detection
    ws_in = prepare_input_for_model(image_bgr, ws_cfg, device)
    ws_out = ws_model([ws_in])[0]

    ws_instances = detector_postprocess(
        ws_out["instances"].to("cpu"), ws_in["height"], ws_in["width"]
    )

    # Class 5 is windshield
    keep = (ws_instances.pred_classes == 5) & (ws_instances.scores >= ws_thresh)
    ws_instances = ws_instances[keep]

    cs_all = []

    # For each windshield, detect stickers inside
    for box in ws_instances.pred_boxes:
        x1, y1, x2, y2 = map(int, box.tolist())
        crop = image_bgr[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        cs_in = prepare_input_for_model(crop, cs_cfg, device)
        cs_out = cs_model([cs_in])[0]

        cs_instances = detector_postprocess(
            cs_out["instances"].to("cpu"), cs_in["height"], cs_in["width"]
        )

        keep_cs = (cs_instances.pred_classes == 4) & (cs_instances.scores >= cs_thresh)
        cs_instances = cs_instances[keep_cs]

        if len(cs_instances) == 0:
            continue

        # Translate boxes to original coords
        cs_boxes = cs_instances.pred_boxes.tensor
        cs_boxes[:, 0::2] += x1
        cs_boxes[:, 1::2] += y1

        inst = Instances(
            image_size=image_bgr.shape[:2],
            pred_boxes=Boxes(cs_boxes),
            scores=cs_instances.scores,
            pred_classes=cs_instances.pred_classes,
        )

        cs_all.append(inst)

    if len(cs_all) > 0:
        return Instances.cat(cs_all)

    # No CS detections
    h, w = image_bgr.shape[:2]
    return create_empty_instances(h, w, device)


@torch.no_grad()
def detect_cs_only(image_bgr, cs_model, cs_cfg, cs_thresh, device):
    """
    Detect stickers on the full image without cropping by windshields.
    
    Steps:
        1. Detect windshields for alignment purposes only.
        2. Detect stickers on the full image.
        3. Determine alignment of each sticker using windshields.
    
    Args:
        image_bgr (numpy.ndarray): Input image in BGR format.
        ws_model (nn.Module): Windshield detection model.
        cs_model (nn.Module): Car sticker detection model.
        ws_cfg (CfgNode): Windshield model config.
        cs_cfg (CfgNode): Sticker model config.
        metadata (MetadataCatalog): Dataset metadata.
        device (str): Device to run model on.
    
    Returns:
        Instances: Sticker Instances object with 'alignments' field added.
    """

    cs_in = prepare_input_for_model(image_bgr, cs_cfg, device)
    cs_out = cs_model([cs_in])[0]

    cs_instances = detector_postprocess(
        cs_out["instances"].to("cpu"), cs_in["height"], cs_in["width"]
    )

    cs_instances = cs_instances[
        (cs_instances.pred_classes == 4) &
        (cs_instances.scores >= cs_thresh)
    ]

    return cs_instances



if __name__ == "__main__":
    args = get_args()

    os.makedirs(args.output_folder, exist_ok=True)
    torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("USING DEVICE:", device)

    # Load models
    ws_model, ws_cfg = load_model(args.ws_config, args.ws_weights, device)
    cs_model, cs_cfg = load_model(args.cs_config, args.cs_weights, device)

    # Load test images
    image_files = [f for f in os.listdir(args.input_folder)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    # Load COCO GT JSON
    with open(args.gt_json) as f:
        gt = json.load(f)

    image_id_lookup = {img["file_name"]: img["id"] for img in gt["images"]}
    category_id_lookup = {cat["name"]: cat["id"] for cat in gt["categories"]}

    predictions = []

    for filename in image_files:
        img_path = os.path.join(args.input_folder, filename)
        image = cv2.imread(img_path)

        if image is None:
            print("Skipping unreadable file:", filename)
            continue

        # Run the chosen mode
        if args.mode == "ws-then-cs":
            cs_instances = detect_ws_then_cs(
                image, ws_model, cs_model, ws_cfg, cs_cfg,
                args.ws_score_thresh, args.cs_score_thresh, device
            )
        else:  # cs-only
            cs_instances = detect_cs_only(
                image, cs_model, cs_cfg,
                args.cs_score_thresh, device
            )

        # Save predictions
        if len(cs_instances) > 0:
            boxes = cs_instances.pred_boxes.tensor.cpu().numpy()
            scores = cs_instances.scores.cpu().numpy()

            for box, score in zip(boxes, scores):
                x1, y1, x2, y2 = box.tolist()
                predictions.append({
                    "image_id": image_id_lookup[filename],
                    "category_id": category_id_lookup["car-sticker"],
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score),
                })

    out_json = os.path.join(args.output_folder, "test_predictions.json")
    with open(out_json, "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"\nSaved predictions to {out_json}\n")
