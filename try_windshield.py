# --- ACRONYMS ---
# ws = windshield
# cs = car stickers

import os
import cv2
import torch
from torchvision import transforms
from fsdet.config import get_cfg
from fsdet.modeling import GeneralizedRCNN
from fsdet.checkpoint import DetectionCheckpointer

input_folder = "datasets/stickers/stickers_ws_test_31shot_1280/"
output_folder = "results/try_windshield_results/" # output to save processed images
os.makedirs(output_folder, exist_ok=True)

torch.cuda.empty_cache()
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # SET GPU HERE
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model(config_path):
    torch.cuda.empty_cache()

    # !!! NOTE: train model first using terminal command !!!
    # CUDA_VISIBLE_DEVICES=2 python3 -m tools.train_net --num-gpus 1 \  --config-file configs/stickers-detection/stickers_2shot_tinyonly_top4_8_random_all_1500iters_lr001_unfreeze_r-nms_fbackbone.yaml
    
    cfg = get_cfg()
    cfg.merge_from_file(config_path) 
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR , "model_final.pth") # Trained model weights
    
    model = GeneralizedRCNN(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.to(device).eval()
    return model

# config separate FSDet for windshield and stickers, adjust config paths accordingly
# best stickers = stickers_ws_31shot_tinyonly_top4_8_random_all_3000iters_lr001_unfreeze_r-nms_fbackbone.yaml
ws_model = load_model("configs/stickers-detection/stickers_ws_31shot_tinyonly_top4_8_random_all_3000iters_lr001_unfreeze_r-nms_fbackbone.yaml")
# cs_model = load_model("configs/stickers-detection/stickers_ws_31shot_tinyonly_top4_8_random_all_3000iters_lr001_unfreeze_r-nms_fbackbone.yaml")

# image to tensor
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((720, 1280)),  # FSDet's input size (H, W)
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0).to(device)

# Get all image files from the folder
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith((".jpg", ".jpeg"))]

print("image files list count ", len(image_files))
for image_filename in image_files:
    image_path = os.path.join(input_folder, image_filename)

    image = cv2.imread(image_path)
    if image is None:
        print(f"Skipping invalid image: {image_path}")
        continue

    image_tensor = preprocess_image(image)

    # detect windshield and store map of its coordinates
    with torch.no_grad():
        windshield_outputs = ws_model(image_tensor)

    # extract detected windshield boxes
    windshield_boxes = windshield_outputs["instances"].pred_boxes.tensor.cpu().numpy()

    # detect stickers within coordinates
    # set windshield coords as new image, then only detect stickers there??
    sticker_detections = []

# display result
