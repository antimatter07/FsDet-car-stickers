import os
import argparse
import shutil
import cv2
import numpy as np
import torch
import json

from fsdet.config import get_cfg
from fsdet.modeling import GeneralizedRCNN
from fsdet.checkpoint import DetectionCheckpointer

import fsdet.data.builtin  # registers datasets


# -------------------------------
# Configs for modification
# -------------------------------

INPUT_FOLDER = "datasets/stickers/stickers_ws_train_31shot_1280/"
INPUT_JSON = "datasets/stickers_split/stickers_ws_train_31shot_1280.json"
OUTPUT_FOLDER = "datasets/cropped_train_data/"
OUTPUT_JSON_FOLDER = "datasets/cropped_train_annot/"


# Utilities

def load_model(config_path: str, weights_path: str, device: torch.device):
    """
    Load a trained FsDet model.

    Args:
        config_path: Path to the model config (.yaml)
        weights_path: Path to the trained weights (.pth)
        device: torch.device to load model on

    Returns:
        A ready-to-use GeneralizedRCNN model
    """
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = weights_path

    model = GeneralizedRCNN(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

    return model.to(device).eval()


def preprocess_image(image_path: str, device: torch.device):
    """
    Convert an image to Detectron2/FsDet input format.

    Args:
        image_path: Path to an image file
        device: torch.device

    Returns:
        A dictionary containing tensor image + metadata
    """
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_tensor = torch.from_numpy(image.astype("float32")).permute(2, 0, 1).to(device)

    return {
        "image": image_tensor,
        "height": image_tensor.shape[1],
        "width": image_tensor.shape[2],
    }


def detect_only_class(image, model, class_id, min_conf=0.7):
    """
    Filter model predictions to include only a specific class with confidence threshold.

    Args:
        image: preprocessed image (dict)
        model: FsDet model
        class_id: target class id
        min_conf: confidence threshold

    Returns:
        Instances filtered by class_id + score
    """
    image_tensor = image["image"].float()

    inputs = {
        "image": image_tensor,
        "height": image["height"],
        "width": image["width"],
    }

    with torch.no_grad():
        outputs = model([inputs])
        instances = outputs[0]["instances"]

    mask = (instances.pred_classes == class_id) & (instances.scores >= min_conf)
    return instances[mask]


def detect_ws(image_filename, image, ws_model, output_folder):
    """
    Detect windshield regions and save cropped images.

    Args:
        image_filename: original filename
        image: processed tensor dict
        ws_model: windshield detection model
        output_folder: where to save cropped WS images

    Returns:
        List of dicts describing cropped images
    """
    ws_instances = detect_only_class(image, ws_model, class_id=5, min_conf=0.7)

    image_np = image["image"].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    json_images = []

    for i, box in enumerate(ws_instances.pred_boxes.tensor):
        x1, y1, x2, y2 = map(int, box.tolist())
        crop = image_np[y1:y2, x1:x2]

        new_name = f"WS_{i}_{image_filename}__.jpg"
        json_images.append({
            "new_filename": new_name,
            "original_filename": image_filename,
            "ws_box": [x1, y1, x2, y2],
            "height": crop.shape[0],
            "width": crop.shape[1],
        })

        cv2.imwrite(os.path.join(output_folder, new_name), crop)

    return json_images


def clear_output_folder(folder_path):
    """
    Delete all contents of a folder safely.

    Args:
        folder_path: target folder
    """
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(path) or os.path.islink(path):
                    os.unlink(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path)
            except Exception as e:
                print(f"Failed to delete {path}: {e}")
    else:
        os.makedirs(folder_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=float, default=0.7)
    args = parser.parse_args()

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_JSON_FOLDER, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on:", device)

    # Load Ws model
    ws_model = load_model(
        "configs/stickers-detection/stickers_ws_31shot_tinyonly_top4_8_random_all_3000iters_lr001_unfreeze_r-nms_fbackbone.yaml",
        "checkpoints/stickers/stickers_ws_31shot_tinyonly_top4_8_random_all_3000iters_lr001_unfreeze_r-nms_fbackbone/model_final.pth",
        device,
    )

    # Get image list
    image_files = [
        f for f in os.listdir(INPUT_FOLDER)
        if f.lower().endswith((".jpg", ".jpeg"))
    ]

    print("> Image file count:", len(image_files))

    clear_output_folder(OUTPUT_FOLDER)
    clear_output_folder(OUTPUT_JSON_FOLDER)
    clear_output_folder("datasets/testing")

    ws_images = []

    # Detect WS 
    for filename in image_files:
        path = os.path.join(INPUT_FOLDER, filename)
        image_tensor = preprocess_image(path, device)
        ws_images.extend(detect_ws(filename, image_tensor, ws_model, OUTPUT_FOLDER))

    print("> Saved cropped windshield images")

    # Load annotations
    with open(INPUT_JSON) as f:
        annot_data = json.load(f)

    original_images = annot_data["images"]
    original_annots = annot_data["annotations"]

    new_images = []
    new_annots = []
    new_img_id = 0
    new_ann_id = 0

    # Rebuild annots
    for ws in ws_images:
        orig_file = ws["original_filename"]
        x_min_ws, y_min_ws, x_max_ws, y_max_ws = ws["ws_box"]

        orig_img = next(img for img in original_images if img["file_name"] == orig_file)
        orig_image_id = orig_img["id"]

        new_images.append({
            "id": new_img_id,
            "file_name": ws["new_filename"],
            "height": ws["height"],
            "width": ws["width"],
        })

        anns_for_img = [a for a in original_annots if a["image_id"] == orig_image_id]

        for i, ann in enumerate(anns_for_img):
            if ann["category_id"] != 91:
                continue

            x, y, w, h = ann["bbox"]

            # Fully inside ws region
            if x < x_min_ws or x + w > x_max_ws or y < y_min_ws or y + h > y_max_ws:
                continue

            # Intersection
            ix1 = max(x, x_min_ws)
            iy1 = max(y, y_min_ws)
            ix2 = min(x + w, x_max_ws)
            iy2 = min(y + h, y_max_ws)

            iw = ix2 - ix1
            ih = iy2 - iy1
            if iw <= 0 or ih <= 0:
                continue

            rel_x = ix1 - x_min_ws
            rel_y = iy1 - y_min_ws

            new_annots.append({
                "id": new_ann_id,
                "image_id": new_img_id,
                "category_id": ann["category_id"],
                "bbox": [rel_x, rel_y, iw, ih],
                "area": iw * ih,
                "iscrowd": ann["iscrowd"],
            })

            new_ann_id += 1

        new_img_id += 1

    final_json = {
        "categories": [
            cat for cat in annot_data["categories"]
            if cat["name"] != "windshield"
        ],
        "images": new_images,
        "annotations": new_annots,
    }

    # Save final annotations
    with open(os.path.join(OUTPUT_JSON_FOLDER, "cropped_annot.json"), "w") as f:
        json.dump(final_json, f)

    print("> Saved adjusted car sticker annotations")


if __name__ == "__main__":
    main()
