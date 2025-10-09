# Preprocess the given dataset and detect all windshields
# Used config: stickers_ws_31shot_tinyonly_top4_8_random_all_3000iters_lr001_unfreeze_r-nms_fbackbone.yaml

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

import fsdet.data.builtin # registers all datasets

# FOR TRAINING DATA
input_folder = "datasets/stickers/stickers_ws_train_10shot_1280/" # train image folder
input_json = "datasets/stickers_split/stickers_ws_train_10shot_1280.json"
output_folder = "datasets/cropped_train_data_10shot/" # where to save cropped images
output_json_folder = "datasets/cropped_train_annot_10shot/" # where to save GT boxes of car stickers

# FOR TEST DATA
# input_folder = "datasets/stickers/stickers_ws_test_31shot_1280/" # test image folder
# input_json = "datasets/stickers/annotations/stickers_ws_31shot_test_1280.json"
# output_folder = "datasets/cropped_test_data/" # where to save cropped images
# output_json_folder = "datasets/cropped_test_annot/" # where to save GT boxes of car stickers


parser = argparse.ArgumentParser(description="Run this file with a minimum confidence score")
parser.add_argument("--conf", type=float, default=0.7, help="minimum confidence score")
args = parser.parse_args()


os.makedirs(output_folder, exist_ok=True)
torch.cuda.empty_cache()


# os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # SET GPU HERE
# device = "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CURRENT DEVICE: ", device)

# Load FsDet model
# This assumes that the model is already trained
def load_model(config_path, weights_path):
    torch.cuda.empty_cache()

    cfg = get_cfg()
    cfg.merge_from_file(config_path) 
    cfg.MODEL.WEIGHTS = weights_path
    
    model = GeneralizedRCNN(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    
    model.to(device).eval()
    return model


ws_model = load_model("configs/stickers-detection/stickers_ws_31shot_tinyonly_top4_8_random_all_3000iters_lr001_unfreeze_r-nms_fbackbone.yaml",
                    "checkpoints/stickers/stickers_ws_31shot_tinyonly_top4_8_random_all_3000iters_lr001_unfreeze_r-nms_fbackbone/model_final.pth")


# converts OpenCV image to tensor for GeneralizedRCNN input format
def preprocess_image(image_path, device):
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


# Filter predictions to only include desired class
# If no confidence is given, only consider those with at least 70% confidence
def detect_only_class(image, model, class_id):

    image_tensor = image["image"]
    if not image_tensor.is_floating_point():
        image_tensor = image_tensor.float()
    
    # image_tensor = image_tensor.to(model_device)

    inputs = {
        "image": image_tensor,
        "height": image["height"],
        "width": image["width"]
    }
    
    with torch.no_grad():
        outputs = model([inputs])
        instances = outputs[0]["instances"]

    conf = min(args.conf, 1.0)
    mask = (instances.pred_classes == class_id) & (instances.scores >= conf)
    filtered_instances = instances[mask]

    return filtered_instances


# detects all windshields in an image
def detect_ws(image_filename, image, ws_model):

    with torch.no_grad():
        ws_instances = detect_only_class(image, ws_model, 5)
    
    image_tensor = image["image"]
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    image_np = (image_np).astype(np.uint8)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    json_images = []
    
    # Crop all predicted windshields
    for i, box in enumerate(ws_instances.pred_boxes.tensor):
        x1, y1, x2, y2 = map(int, box.tolist())
        crop = image_np[y1:y2, x1:x2]

        json_images.append({
        "new_filename": f"WS_{i}_{image_filename}__.jpg",
        "original_filename": image_filename,
        "ws_box": [x1, y1, x2, y2],
        "height": crop.shape[0],
        "width": crop.shape[1],
        })

        # Save the cropped image to output_folder
        output_path =  f"{output_folder}WS_{i}_{image_filename}__.jpg"
        cv2.imwrite(output_path, crop)
    
    # return json format of the cropped WS images
    return json_images

# Delete all files and subfolders and the ouput folder
def clear_output_folder(folder_path):
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path) 
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        os.makedirs(folder_path)


# Get all image files from the folder
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith((".jpg", ".jpeg"))]
print("> Image file count: ", len(image_files))

clear_output_folder(output_folder)
clear_output_folder(output_json_folder)
clear_output_folder("datasets/testing")

ws_images = []

for image_filename in image_files:
    image_path = os.path.join(input_folder, image_filename)
    image_tensor = preprocess_image(image_path, device)

    ws_images.extend(detect_ws(image_filename, image_tensor, ws_model))


print(f"> Saved windshield images to {output_folder}")


# Get all the data annotations
with open(input_json) as f:
    annot_data = json.load(f)

original_images = annot_data["images"]
original_annots = annot_data["annotations"]

new_images = []
new_annotations = []
new_image_id = 0
new_annotation_id = 0

for ws_image in ws_images:
    original_filename = ws_image["original_filename"]
    ws_box = ws_image["ws_box"]
    x_min_ws, y_min_ws, x_max_ws, y_max_ws = ws_box

    # Find original image_id
    original_img = next(img for img in original_images if img["file_name"] == original_filename)
    orig_image_id = original_img["id"]

    # Add new image entry
    new_images.append({
        "id": new_image_id,
        "file_name": ws_image["new_filename"],
        "height": ws_image["height"],
        "width": ws_image["width"],
    })

    # Filter annotations for this image
    anns_for_image = [ann for ann in original_annots if ann["image_id"] == orig_image_id]

    # for ann in anns_for_image:
    for i, ann in enumerate(anns_for_image):
        x, y, w, h = ann["bbox"]

        # Check if the bbox is a sticker (category_id=91 || category_id=1)
        if ann["category_id"] != 91:
            continue 
        
        # Check if the sticker is within the WS region
        # includes every sticker GT box that overlap with the cropped windshield
        # if x + w < x_min_ws or x > x_max_ws or y + h < y_min_ws or y > y_max_ws:
        #     continue
        
        # excludes any sticker GT box that goes outside the cropped windshield
        if x < x_min_ws or x + w > x_max_ws or y < y_min_ws or y + h > y_max_ws:
            print("car sticker outside windshield")
            continue

        # Check for clipping / intersection box
        inter_x1 = max(x, x_min_ws)
        inter_y1 = max(y, y_min_ws)
        inter_x2 = min(x + w, x_max_ws)
        inter_y2 = min(y + h, y_max_ws)

        inter_w = inter_x2 - inter_x1
        inter_h = inter_y2 - inter_y1

        if inter_w <= 0 or inter_h <= 0:
            continue

        # Adjust bbox relative to WS crop
        rel_x = inter_x1 - x_min_ws
        rel_y = inter_y1 - y_min_ws

        new_bbox = [rel_x, rel_y, inter_w, inter_h]
        new_area = inter_w * inter_h

        new_annotations.append({
            "id": new_annotation_id,
            "image_id": new_image_id,
            "category_id": ann["category_id"],
            "bbox": new_bbox,
            "area": new_area,
            "iscrowd": ann["iscrowd"],
        })
        
        # -- TBD (adjusted annotations visual) --
        img = cv2.imread(f"{output_folder}{ws_image['new_filename']}")
        cv2.rectangle(img, (rel_x, rel_y), (rel_x + inter_w, rel_y + inter_h), (0, 255, 0), 2) 
        path = f"datasets/testing/_{i}_{original_filename}.jpg"
        cv2.imwrite(path, img)
        # -- END TBD --
        
        new_annotation_id += 1

    new_image_id += 1


adjusted_json = {
    "categories": [cat for cat in annot_data["categories"] if cat["name"] != "windshield"], # remove windshield
    "images": new_images,
    "annotations": new_annotations
}

# Save adjusted annotations
with open(f"{output_json_folder}cropped_annot.json", "w") as f:
    json.dump(adjusted_json, f)


print(f"> Saved adjusted car sticker annotations to {output_folder}")
