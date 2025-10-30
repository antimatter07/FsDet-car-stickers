"""
test_drawpreds_stickersonly.py

Visualize sticker predictions on test set of each k-shot split for sticker only (no windshield-guided pipeline yet)

This script implements performs inference with a sticker detector trained on sticker annotations on the full-image and
draws predictions made by the model with Detectron2's visualizer in the output folder.

"""


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
import shutil

from detectron2.utils.visualizer import Visualizer
import detectron2.data.transforms as T

import fsdet.data.builtin # registers all datasets


# Test the FULL IMAGES
#input_folder = "datasets/stickers/stickers_ws_test_31shot_1280/" # test image folder
output_folder = "results/stickers_only/predictions_visualization/31_shot/" # output to save processed images
# NEED to set dataset name for retrieving the test set data
dataset_name = "stickers_31shot_1280_test_tinyonly_top4" # registered name of the test dataset
input_folder = MetadataCatalog.get(dataset_name).image_root

STICKERS_SCORE_THRESHOLD = 0.0

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


# For 31-shot
# fsdet/FsDet-car-stickers/configs/stickers-detection/stickers_only_31shot.yaml
# fsdet/FsDet-car-stickers/results/stickers_only/31shot/best/model_0000999.pth

# For 10-shot

# For 5-shot

# For 2-shot
cs_model, cs_cfg = load_model(
    "configs/stickers-detection/stickers_only_31shot.yaml",
    "results/stickers_only/31shot/best/model_0000999.pth"
)


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
def detect_cs(image_bgr, cs_model, cs_cfg, metadata, device="cuda"):

     # Required pre-processing for image
    cs_inputs = prepare_input_for_model(image_bgr, cs_cfg, device)
    cs_outputs = cs_model([cs_inputs])[0]

    cs_instances = detector_postprocess(
            cs_outputs["instances"].to("cpu"), cs_inputs["height"], cs_inputs["width"]
    )

    keep = (cs_instances.pred_classes == 4) & (cs_instances.scores >= STICKERS_SCORE_THRESHOLD)
    cs_instances = cs_instances[keep]
    
    # Combine all sticker predictions
    if len(cs_instances) == 0:
        # Create empty instances with all fields defined
        height, width = image_bgr.shape[:2]
        cs_instances = create_empty_instances(height, width)
    return cs_instances
        



def clean_instance_fields(instances):
    """
    Remove unnecessary tensor labels (pred_masks, etc.) to make visualization cleaner.
    """
    # --- Clone safely (works in older Detectron2/FsDet) ---
    new_instances = Instances(instances.image_size)
    for k, v in instances.get_fields().items():
        new_instances.set(k, v.clone() if torch.is_tensor(v) else v)

    # Remove mask/tensor fields that clutter visualization
    for field in ["pred_masks", "gt_boxes", "gt_classes"]:
        if hasattr(new_instances, field):
            delattr(new_instances, field)

    return new_instances
    

def visualize_result(image_path, cs_instances):
    image_bgr = cv2.imread(image_path)
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    metadata = MetadataCatalog.get(dataset_name)
    metadata.thing_classes = ["class0", "class1", "class2", "class3", "sticker"]

    # move to CPU for Visualizer
    cs_instances_cpu = clean_instance_fields(cs_instances.to("cpu"))



    # visualize resultsfcv2.im
    vis = Visualizer(image.copy(), metadata=metadata, scale=1.0)
    output = vis.draw_instance_predictions(cs_instances_cpu)

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
 
    image_bgr = cv2.imread(image_path)
    cs_instances = detect_cs(
    image_bgr, cs_model, cs_cfg, MetadataCatalog.get(dataset_name), device=device
    )


    #ws_instances, cs_instances = detect_ws_then_cs(image_tensor, ws_model, cs_model, MetadataCatalog.get(dataset_name))

    visualize_result(image_path, cs_instances) 

