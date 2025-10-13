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

from detectron2.utils.visualizer import Visualizer

import fsdet.data.builtin # registers all datasets


# Test the FULL IMAGES
input_folder = "datasets/stickers/stickers_ws_test_31shot_1280/" # test image folder
output_folder = "results/test_wscs_results/" # output to save processed images
dataset_name = "stickers_ws_31shot_1280_test_tinyonly_top4" # registered name of the test dataset

os.makedirs(output_folder, exist_ok=True)
torch.cuda.empty_cache()

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # SET GPU HERE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CURRENT DEVICE: ", device)

# Load FsDet model
def load_model(config_path, weights_path):
    torch.cuda.empty_cache()

    # !!! NOTE: train model first using terminal command !!!
    cfg = get_cfg()
    cfg.merge_from_file(config_path) 
    cfg.MODEL.WEIGHTS = weights_path
    
    model = GeneralizedRCNN(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    
    model.to(device).eval()
    return model


# best ws+stickers config : stickers_ws_31shot_tinyonly_top4_8_random_all_3000iters_lr001_unfreeze_r-nms_fbackbone.yaml
#TODO: get temporary weights
ws_model = load_model("configs/stickers-detection/stickers_ws_31shot_tinyonly_top4_8_random_all_3000iters_lr001_unfreeze_r-nms_fbackbone.yaml",
                    "checkpoints/stickers/stickers_ws_31shot_tinyonly_top4_8_random_all_3000iters_lr001_unfreeze_r-nms_fbackbone/model_final.pth")
cs_model = load_model("configs/stickers-detection/test_predicts/ws_then_cs_testpredicts.yaml", "datasets/testing/model_final.pth") # !! THIS ONE LOL


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
def detect_only_class(img, model, class_id):
    with torch.no_grad():
        outputs = model([img])
        instances = outputs[0]["instances"].to(device)

    # only consider those with at least 70% confidence
    mask = (instances.pred_classes == class_id) & (instances.scores >= 0.7) 
    filtered_instances = instances[mask]

    return filtered_instances


def create_empty_instances(height, width):
    empty_instances = Instances((height, width))
    empty_instances.pred_boxes = Boxes(torch.empty((0, 4)).to(device))
    empty_instances.scores = torch.empty((0,)).to(device)
    empty_instances.pred_classes = torch.empty((0,), dtype=torch.int64).to(device)

    return empty_instances


def detect_ws_then_cs(img, ws_model, cs_model, metadata):
    # Run inference for windshields
    # with torch.no_grad():
    #     ws_outputs = ws_model([img])
    #     ws_instances = ws_outputs[0]["instances"]
    ws_instances = detect_only_class(img, ws_model, 5)
    
    # Prepare the image for cropping
    image_tensor = img["image"]
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    image_np = (image_np * 255).astype(np.uint8)

    all_cs_instances = []

    # Loop over all predicted windshields
    for i, box in enumerate(ws_instances.pred_boxes.tensor):
        x1, y1, x2, y2 = map(int, box.tolist())
        print(f"Processing WS box {i}: ({x1}, {y1}, {x2}, {y2})")

        # Crop the region
        crop = image_np[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        # Prepare input dict for sticker model
        crop_tensor = torch.from_numpy(crop_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).to(device)
        cs_input = {
            "image": crop_tensor,
            "height": crop_tensor.shape[1],
            "width": crop_tensor.shape[2],
        }

        # Run sticker detector on the cropped windshield
        # with torch.no_grad():
        #     cs_outputs = cs_model([cs_input])
        #     cs_instances = cs_outputs[0]["instances"]
        cs_instances = detect_only_class(cs_input, cs_model, 4)

        if len(cs_instances) == 0:
            # print(f"No stickers detected in windshield box {i}")
            continue

        # Offset bounding boxes to original coordinates
        offset = torch.tensor([x1, y1, x1, y1], device=cs_instances.pred_boxes.device)
        cs_instances.pred_boxes.tensor += offset

        all_cs_instances.append(cs_instances)

        print(f"Detected {len(cs_instances)} car stickers in windshield box {i}.")
    
        for j, box in enumerate(cs_instances.pred_boxes.tensor):
            cx1, cy1, cx2, cy2 = box.tolist()
            print(f"Sticker Box {j}: x1={cx1:.2f}, y1={cy1:.2f}, x2={cx2:.2f}, y2={cy2:.2f}")

    if all_cs_instances:
        combined_cs_instances = Instances.cat(all_cs_instances)
    else:
        combined_cs_instances = create_empty_instances(img["image"].shape[1], img["image"].shape[2])

    return ws_instances, combined_cs_instances


def visualize_result(image_path, ws_instances, cs_instances):
    image_bgr = cv2.imread(image_path)
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    metadata = MetadataCatalog.get(dataset_name)
    #metadata.thing_classes == ["windshield", "car-sticker"] #TODO: get classes from model

    # move to CPU for Visualizer
    ws_instances_cpu = ws_instances.to("cpu")
    cs_instances_cpu = cs_instances.to("cpu")

    # combine all instances
    all_instances_cpu = Instances.cat([ws_instances_cpu, cs_instances_cpu])

    # visualize resultsfcv2.im
    vis = Visualizer(image.copy(), metadata=metadata, scale=1.0)
    output = vis.draw_instance_predictions(all_instances_cpu)

    # Save the image in output folder
    output_image = output.get_image()
    output_path = output_folder + os.path.split(image_path)[1] + "_output.jpg"
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
    image_tensor = preprocess_image(image_path)

    ws_instances, cs_instances = detect_ws_then_cs(image_tensor, ws_model, cs_model, MetadataCatalog.get(dataset_name))

    visualize_result(image_path, ws_instances, cs_instances) 
    
# TODO: Display accuracy result
