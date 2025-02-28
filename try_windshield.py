# --- ACRONYMS ---
# ws = windshield
# cs = car stickers

import os
import cv2
import torch
from torchvision import transforms
from fsdet.modeling import GeneralizedRCNN
from fsdet.checkpoint import DetectionCheckpointer

input_folder = "datasets/stickers/stickers_ws_test_31shot_1280/"
output_folder = "results/try_windshield_results/" # output to save processed images
os.makedirs(output_folder, exist_ok=True)

torch.cuda.set_device(2)
print("Current GPU: ", torch.cuda.current_device())

def load_model(config_path, device="cuda:2" if torch.cuda.is_available() else "cpu"):
    torch.cuda.empty_cache()

    # !!! NOTE: train model first using terminal command !!!
    # CUDA_VISIBLE_DEVICES=2 python3 -m tools.train_net --num-gpus 1 \  --config-file configs/stickers-detection/stickers_2shot_tinyonly_top4_8_random_all_1500iters_lr001_unfreeze_r-nms_fbackbone.yaml
    
    cfg = get_cfg()
    # cfg.set_new_allowed(True)
    cfg.merge_from_file(config_path) 
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR , "model_final.pth") # Trained model weights
    
    model = GeneralizedRCNN(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()
    return model

# config separate FSDet for windshield and stickers, adjust config paths accordingly
# best stickers = stickers_ws_31shot_tinyonly_top4_8_random_all_3000iters_lr001_unfreeze_r-nms_fbackbone.yaml
    # TODO: make test yaml file with only 300 iters for trial 
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

# for image_filename in image_files:
#     image_path = os.path.join(input_folder, image_filename)

#     image = cv2.imread(image_path)
#     if image is None:
#         print(f"Skipping invalid image: {image_path}")
#         continue

#     image_tensor = preprocess_image(image)

#     print("TENSOR: ", image_tensor)

#     break

#     # detect windshield and store map of its coordinates
#     # with torch.no_grad():
#     #     print("Detecting windshields...")
#     #     windshield_outputs = ws_model(image_tensor)
#     #     print("Windshields done")

#     # detect stickers within coordinates
#     # set windshield coords as new image, then only detect stickers there??

# # display result

