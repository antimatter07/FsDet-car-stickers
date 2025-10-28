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
import shutil
from detectron2.utils.visualizer import Visualizer
import detectron2.data.transforms as T
import fsdet.data.builtin  # registers all datasets
from scipy.ndimage import gaussian_filter

# --- PATH CONFIG ---
input_folder = "datasets/stickers/stickers_ws_test_31shot_1280/"
output_folder = "results/test_wscs_10shot_images_with_predictions/"
dataset_name = "stickers_ws_31shot_1280_test_tinyonly_top4"

# confidence thresholds
WS_SCORE_THRESHOLD = 0.7
STICKERS_SCORE_THRESHOLD = 0.0

os.makedirs(output_folder, exist_ok=True)
torch.cuda.empty_cache()
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CURRENT DEVICE:", device)


# --- MODEL LOADING ---
def load_model(config_path, weights_path):
    torch.cuda.empty_cache()
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = weights_path

    model = GeneralizedRCNN(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.to(device).eval()
    return model, cfg


ws_model, ws_cfg = load_model(
    "configs/stickers-detection/stickers_ws_31shot_tinyonly_top4_8_random_all_3000iters_lr001_unfreeze_r-nms_fbackbone.yaml",
    "checkpoints/stickers/stickers_ws_31shot_tinyonly_top4_8_random_all_3000iters_lr001_unfreeze_r-nms_fbackbone/model_final.pth"
)

cs_model, cs_cfg = load_model(
    "configs/stickers-detection/ws_then_cs_10shot.yaml",
    "results/ws_then_cs_10shot/model_0001799.pth"
)


# --- HELPERS ---
def prepare_input_for_model(image_bgr, cfg, device):
    transform_gen = T.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST],
        cfg.INPUT.MAX_SIZE_TEST,
    )
    height, width = image_bgr.shape[:2]
    image_resized = transform_gen.get_transform(image_bgr).apply_image(image_bgr)
    image_tensor = torch.as_tensor(image_resized.astype("float32").transpose(2, 0, 1))
    return {"image": image_tensor.to(device), "height": height, "width": width}


def create_empty_instances(height, width):
    empty = Instances((height, width))
    empty.pred_boxes = Boxes(torch.empty((0, 4)).to(device))
    empty.scores = torch.empty((0,)).to(device)
    empty.pred_classes = torch.empty((0,), dtype=torch.int64).to(device)
    return empty


# --- DETECTION PIPELINE ---
@torch.no_grad()
def detect_ws_then_cs(image_bgr, ws_model, cs_model, ws_cfg, cs_cfg, metadata, device="cuda"):
    ws_inputs = prepare_input_for_model(image_bgr, ws_cfg, device)
    ws_outputs = ws_model([ws_inputs])[0]
    ws_instances = detector_postprocess(ws_outputs["instances"].to("cpu"), ws_inputs["height"], ws_inputs["width"])

    keep = (ws_instances.pred_classes == 5) & (ws_instances.scores >= WS_SCORE_THRESHOLD)
    ws_instances = ws_instances[keep]

    cs_instances_all = []
    for box in ws_instances.pred_boxes:
        x1, y1, x2, y2 = map(int, box.tolist())
        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        cs_inputs = prepare_input_for_model(crop, cs_cfg, device)
        cs_outputs = cs_model([cs_inputs])[0]
        cs_instances = detector_postprocess(cs_outputs["instances"].to("cpu"), cs_inputs["height"], cs_inputs["width"])
        keep = (cs_instances.pred_classes == 4) & (cs_instances.scores >= STICKERS_SCORE_THRESHOLD)
        cs_instances = cs_instances[keep]

        if len(cs_instances) > 0:
            cs_boxes = cs_instances.pred_boxes.tensor
            cs_boxes[:, 0::2] += x1
            cs_boxes[:, 1::2] += y1
            shifted = Instances(image_size=image_bgr.shape[:2])
            shifted.pred_boxes = Boxes(cs_boxes)
            shifted.scores = cs_instances.scores
            shifted.pred_classes = cs_instances.pred_classes
            cs_instances_all.append(shifted)

    if len(cs_instances_all) > 0:
        cs_instances = Instances.cat(cs_instances_all)
    else:
        cs_instances = create_empty_instances(*image_bgr.shape[:2])
    return ws_instances, cs_instances


# --- HEATMAP ACCUMULATION ---
def accumulate_heatmap(cs_instances, heatmap):
    for box in cs_instances.pred_boxes.tensor.cpu().numpy():
        x1, y1, x2, y2 = box.astype(int)
        x1 = np.clip(x1, 0, heatmap.shape[1] - 1)
        y1 = np.clip(y1, 0, heatmap.shape[0] - 1)
        x2 = np.clip(x2, 0, heatmap.shape[1] - 1)
        y2 = np.clip(y2, 0, heatmap.shape[0] - 1)
        heatmap[y1:y2, x1:x2] += 1
    return heatmap


# --- CLEAR OUTPUT FOLDER ---
def clear_output_folder():
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)


clear_output_folder()


# --- MAIN LOOP ---
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
print("> Found", len(image_files), "images.")

# initialize heatmap with shape from the first image
sample_img = cv2.imread(os.path.join(input_folder, image_files[0]))
heatmap = np.zeros(sample_img.shape[:2], dtype=np.float32)

for idx, img_name in enumerate(image_files):
    image_path = os.path.join(input_folder, img_name)
    image_bgr = cv2.imread(image_path)
    ws_instances, cs_instances = detect_ws_then_cs(image_bgr, ws_model, cs_model, ws_cfg, cs_cfg, MetadataCatalog.get(dataset_name), device=device)
    heatmap = accumulate_heatmap(cs_instances, heatmap)

    if (idx + 1) % 5 == 0:
        print(f"Processed {idx + 1}/{len(image_files)} images...")

# --- POSTPROCESS HEATMAP ---
# normalize to [0, 255]
heatmap_normalized = (255 * (heatmap / heatmap.max())).astype(np.uint8)
heatmap_blurred = gaussian_filter(heatmap_normalized, sigma=25)

# Save raw heatmap
np.save(os.path.join(output_folder, "sticker_heatmap.npy"), heatmap)
print("Saved raw heatmap array.")

# --- OVERLAY HEATMAP ON SAMPLE IMAGE ---
color_map = cv2.applyColorMap(heatmap_blurred, cv2.COLORMAP_JET)
overlay = cv2.addWeighted(sample_img, 0.5, color_map, 0.5, 0)
cv2.imwrite(os.path.join(output_folder, "sticker_heatmap_overlay.png"), overlay)
print("Saved heatmap overlay image to:", os.path.join(output_folder, "sticker_heatmap_overlay.png"))
