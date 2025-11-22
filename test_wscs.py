# --- ACRONYMS ---
# ws = windshield
# cs = car stickers
import json
import os
import cv2
import numpy
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


# Load FsDet model
def load_model(config_path, weights_path):
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
    torch.cuda.empty_cache()

    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = weights_path

    model = GeneralizedRCNN(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.to(device).eval()

    return model, cfg



def create_empty_instances(height, width):
    """
    Create an empty Instances object with the required fields for predictions.
    
    Args:
        height (int): Image height.
        width (int): Image width.
    
    Returns:
        Instances: Empty Instances object with pred_boxes, scores, pred_classes initialized.
    """
    empty_instances = Instances((height, width))
    empty_instances.pred_boxes = Boxes(torch.empty((0, 4)).to(device))
    empty_instances.scores = torch.empty((0,)).to(device)
    empty_instances.pred_classes = torch.empty((0,), dtype=torch.int64).to(device)

    return empty_instances


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


# determines the alignment of a sticker prediction
def get_alignment(ws_box, sticker_box):
    """
    Determine alignment ('left' or 'right') of a sticker relative to a windshield box.
    
    Args:
        ws_box (list[float]): Windshield bounding box [x1, y1, x2, y2].
        sticker_box (list[float]): Sticker bounding box [x1, y1, x2, y2].
    
    Returns:
        str: "left" or "right" depending on relative position of sticker to windshield center.
    """
    wx1, wy1, wx2, wy2 = ws_box
    sx1, sy1, sx2, sy2 = sticker_box
    
    w_center = (wx1 + wx2) / 2.0
    s_center = (sx1 + sx2) / 2.0

    return "left" if s_center < w_center else "right"


@torch.no_grad()
def detect_ws_then_cs(image_bgr, ws_model, cs_model, ws_cfg, cs_cfg, metadata, device="cuda"):
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

    # For each windshield box, crop and detect stickers 
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

        # Shift sticker boxes to full image coordinates 
        if len(cs_instances) > 0:
            cs_boxes = cs_instances.pred_boxes.tensor
            cs_boxes[:, 0::2] += x1  # shift x1, x2
            cs_boxes[:, 1::2] += y1  # shift y1, y2

            # Determine if sticker is left or right aligned
            alignments = [get_alignment(b, box.tolist()) for b in cs_boxes.tolist()]
                
            shifted_instances = Instances(
                image_size=image_bgr.shape[:2],
                pred_boxes=Boxes(cs_boxes),
                scores=cs_instances.scores,
                pred_classes=cs_instances.pred_classes,
            )
            shifted_instances.set("alignments", alignments)
            cs_instances_all.append(shifted_instances)

    #  Combine all sticker detections 
    if len(cs_instances_all) > 0:
        cs_instances = Instances.cat(cs_instances_all)
    else:
        # Create empty instances with all fields defined
        height, width = image_bgr.shape[:2]
        cs_instances = create_empty_instances(height, width)

    return ws_instances, cs_instances


@torch.no_grad()
def detect_cs_only(image_bgr, ws_model, cs_model, ws_cfg, cs_cfg, metadata, device="cuda"):
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

    H, W = image_bgr.shape[:2]

    # ws detection
    ws_inputs = prepare_input_for_model(image_bgr, ws_cfg, device)
    ws_outputs = ws_model([ws_inputs])[0]

    ws_instances = detector_postprocess(
        ws_outputs["instances"].to("cpu"), ws_inputs["height"], ws_inputs["width"]
    )

    # Filter windshields
    keep = (ws_instances.pred_classes == 5) & (ws_instances.scores >= WS_SCORE_THRESHOLD)
    ws_instances = ws_instances[keep]


    # Sticker detection (full frame, no crop)
    cs_inputs = prepare_input_for_model(image_bgr, cs_cfg, device)
    cs_outputs = cs_model([cs_inputs])[0]

    cs_instances = detector_postprocess(
        cs_outputs["instances"].to("cpu"), cs_inputs["height"], cs_inputs["width"]
    )

    keep = (cs_instances.pred_classes == 4) & (cs_instances.scores >= STICKERS_SCORE_THRESHOLD)
    cs_instances = cs_instances[keep]

    # Determine alignment
    num_stickers = len(cs_instances)
    alignments = []
    if num_stickers > 0:
        if len(ws_instances) > 0:
            ws_box = ws_instances.pred_boxes.tensor[0].tolist()

            sticker_boxes = cs_instances.pred_boxes.tensor.tolist()
            alignments = [get_alignment(ws_box, sbox) for sbox in sticker_boxes]
        else:
            alignments = ["unknown"] * num_stickers

    cs_instances.set("alignments", alignments)

    return cs_instances


def clean_instance_fields(instances):
    """
    Clean unnecessary fields from Instances object for visualization.
    
    Args:
        instances (Instances): Instances object to clean.
    
    Returns:
        Instances: Cleaned Instances object without mask/GT fields.
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
    

def visualize_result(image_path, ws_instances, cs_instances):
    """
    Visualize detected windshields and stickers on an image and save the output.
    
    Args:
        image_path (str): Path to input image.
        ws_instances (Instances): Detected windshield instances.
        cs_instances (Instances): Detected sticker instances.
    """
    image_bgr = cv2.imread(image_path)
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    metadata = MetadataCatalog.get(dataset_name)
    metadata.thing_classes = ["class0", "class1", "class2", "class3", "sticker", "windshield"]

    # move to CPU for Visualizer
    ws_instances_cpu = clean_instance_fields(ws_instances.to("cpu"))
    cs_instances_cpu = clean_instance_fields(cs_instances.to("cpu"))

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
    """
    Delete all files and subfolders in the output folder.
    Creates the folder if it does not exist.
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
    


if __name__ == "__main__":

    # Test on full images
    input_folder = "datasets/stickers/stickers_ws_test_31shot_1280/" # test image folder
    output_folder = "results/stickers_only/2shot/test_predictions/" # output to save json predictions
    dataset_name = "stickers_ws_31shot_1280_test_tinyonly_top4" # registered name of the test dataset
    gt_json = "datasets/stickers/annotations/stickers_ws_31shot_test_1280.json" # only used to convert prediction filenames to image_id
    
    
    #confidence score threshold for each classs
    WS_SCORE_THRESHOLD = 0.7
    STICKERS_SCORE_THRESHOLD = 0.05
    
    os.makedirs(output_folder, exist_ok=True)
    torch.cuda.empty_cache()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # SET GPU HERE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CURRENT DEVICE: ", device)
    
    
    # best ws+stickers config : stickers_ws_31shot_tinyonly_top4_8_random_all_3000iters_lr001_unfreeze_r-nms_fbackbone.yaml
    # Weights for best 31shot: results/ws_then_cs_31shot/model_0000299.pth
    # Weights for best 2shot: results/ws_then_cs_2shot/model_0001799.pth
    # Weights for best 5shot: results/ws_then_cs_5shot/model_0001399.pth
    # weights for best 10shot: results/ws_then_cs_10shot/model_0001799.pth
    ws_model, ws_cfg = load_model(
        "configs/stickers-detection/stickers_ws_31shot_tinyonly_top4_8_random_all_3000iters_lr001_unfreeze_r-nms_fbackbone.yaml",
        "checkpoints/stickers/stickers_ws_31shot_tinyonly_top4_8_random_all_3000iters_lr001_unfreeze_r-nms_fbackbone/model_final.pth"
    )
    
    # cs_model, cs_cfg = load_model(
    #     "configs/stickers-detection/ws_then_cs_31shot.yaml",
    #     "results/ws_then_cs_31shot/model_0000299.pth"
    # )
    
    cs_model, cs_cfg = load_model(
        "configs/stickers-detection/stickers_only_2shot.yaml",
        "results/stickers_only/2shot/best/model_0001799.pth"
    )

    clear_output_folder()


    # Get all image files from the folder
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith((".jpg", ".jpeg"))]
    print("> Image file count: ", len(image_files))
    print("> Dataset name: ", dataset_name)
    
    
    # Create lookup for filename to image_id
    with open(gt_json) as f:
        gt = json.load(f)
    
    image_id_lookup = { 
        image["file_name"] : image["id"] for image in gt["images"]
    }
    
    category_id_lookup = {
        category["name"] : category["id"] for category in gt["categories"]
    }
    
    predictions = []
    
    for image_filename in image_files:
        image_path = os.path.join(input_folder, image_filename)
        image_bgr = cv2.imread(image_path)
        
        # ws_instances, cs_instances = detect_ws_then_cs(
        # image_bgr, ws_model, cs_model, ws_cfg, cs_cfg, MetadataCatalog.get(dataset_name), device=device
        # )
        cs_instances = detect_cs_only(
           image_bgr, ws_model, cs_model, ws_cfg, cs_cfg, MetadataCatalog.get(dataset_name), device=device 
        )
    
        if len(cs_instances) > 0:
            boxes = cs_instances.pred_boxes.tensor.cpu().numpy()
            scores = cs_instances.scores.cpu().numpy()
            labels = cs_instances.pred_classes.cpu().numpy()
            alignments = cs_instances.alignments
            
            for box, score, label, alignment in zip(boxes, scores, labels, alignments):
                x1, y1, x2, y2 = box.tolist()
                width, height = x2 - x1, y2 - y1
                category_id = category_id_lookup["car-sticker"]
    
                predictions.append({
                    "image_id": image_id_lookup[image_filename],
                    "category_id": category_id,
                    "bbox": [x1, y1, width, height],
                    "score": float(score),
                    "alignment": alignment
                })
        # visualize_result(image_path, ws_instances, cs_instances)
    
    predictions_json_path = output_folder + "test_predictions.json"
    with open(predictions_json_path, "w") as f:
        json.dump(predictions, f, indent=2)
    
    print(f"Saved predictions to {predictions_json_path}")

