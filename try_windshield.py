# --- ACRONYMS ---
# ws = windshield
# cs = car stickers

import os
import cv2
import torch
from torchvision import transforms
from PIL import Image
from fsdet.config import get_cfg
from fsdet.modeling import GeneralizedRCNN
from fsdet.checkpoint import DetectionCheckpointer
from fsdet.evaluation import stickers_evaluation
from detectron2.data import MetadataCatalog, DatasetCatalog, detection_utils as utils
from fsdet.data import builtin


input_folder = "datasets/stickers/stickers_ws_test_31shot_1280/" # test image folder
output_folder = "results/try_windshield_results/" # output to save processed images
os.makedirs(output_folder, exist_ok=True)

torch.cuda.empty_cache()
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # SET GPU HERE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CURRENT DEVICE: ", device)

def load_model(config_path):
    torch.cuda.empty_cache()

    # !!! NOTE: train model first using terminal command !!!
    cfg = get_cfg()
    cfg.merge_from_file(config_path) 
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR , "model_final.pth") # Trained model weights
    
    model = GeneralizedRCNN(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    checkpoint = torch.load(cfg.MODEL.WEIGHTS, map_location="cpu")

    model.to(device).eval()
    return model

# best ws+stickers config : stickers_ws_31shot_tinyonly_top4_8_random_all_3000iters_lr001_unfreeze_r-nms_fbackbone.yaml
ws_model = load_model("configs/stickers-detection/stickers_ws_31shot_tinyonly_top4_8_random_all_3000iters_lr001_unfreeze_r-nms_fbackbone.yaml")
# cs_model = load_model("configs/stickers-detection/stickers_ws_31shot_tinyonly_top4_8_random_all_3000iters_lr001_unfreeze_r-nms_fbackbone.yaml")

# converts OpenCV image to tensor for GeneralizedRCNN input format
def preprocess_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_tensor = torch.as_tensor(img_rgb.transpose(2, 0, 1).astype("flot32"))

    inputs = {
        "image": img_tensor,
        "height": img_tensor.shape[1],
        "width": img_tensor.shape[2],
    }

    inputs["image"] = inputs["image"].to(device)

    # cfg = get_cfg()
    
    # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image_pil = Image.fromarray(image_rgb)

    # print("!!mean, std: ", cfg.MODEL.PIXEL_MEAN, ", ", cfg.MODEL.PIXEL_STD)
    # transform = transforms.Compose([
    #     transforms.Resize((720, 1280)),  # FSDet's input size (H, W)
    #     transforms.ToTensor(),
    #     transforms.Normalize(cfg.MODEL.PIXEL_MEAN, cfg.MODEL.PIXEL_STD)
    # ])
    # image_tensor = transform(image_pil).to(device)
    # return [{"image": image_tensor}]

dataset_name = "tinyonly_top4_stickers_ws_31shot_1280"

# Get all image files from the folder
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith((".jpg", ".jpeg"))]
print("Image file count: ", len(image_files))


for image_filename in image_files:
    image_path = os.path.join(input_folder, image_filename)
    
    image = cv2.imread(image_path)
    print("INPUT IMAGE: ", image_path)
    
    if image is None:
        print(f"Skipping invalid image: {image_path}")
        continue

    image_tensor = preprocess_image(image)
    
    # detect windshield and store map of its coordinates
    with torch.no_grad():
        windshield_outputs = ws_model(image_tensor)

    # extract detected windshield boxes
    if windshield_outputs:
        instances = windshield_outputs[0].get("instances", None)
        print("INSTANCE:")
        print(instances)
    #     if instances and hasattr(instances, "pred_boxes"):
    #         print("Windshield detected")
    #         windshield_boxes = instances.pred_boxes.tensor.cpu().numpy()
    #     else:
    #         print("No windshields detected")
    #         windshield_boxes = []
    # else:
    #     windshield_boxes = []
    
    break
    

    # # detect stickers within coordinates
    # sticker_detections = []
    # for x1, y1, x2, y2 in windshield_boxes:
    #     x1, y1, x2, y2 = map( int, [x1, y1, x2, y2])

    #     windshield_region = image[y1:y2, x1:x2].copy()
    #     windshield_tensor = preprocess_image(windshield_crop)

    #     with torch.no_grad():
    #         sticker_outputs = sticker_model(windshield_tensor)

    #     if "instances" in sticker_outputs:
    #         sticker_boxes = sticker_outputs["instances"].pred_boxes.tensor.cpu().numpy()

    #         for sx1, sy1, sx2, sy2 in sticker_boxes:
    #             sx1, sy1, sx2, sy2 = map(int, [sx1, sy1, sx2, sy2])

    #             # convert sticker coordinates to original image
    #             global_sx1, global_sy1 = x1 + sx1, y1 + sy1
    #             global_sx2, global_sy2 = x1 + sx2, y1 + sy2

    #             # store sticker detections
    #             sticker_detections.append((global_sx1, global_sy1, global_sx2, global_sy2))

    #             # Draw sticker detection
    #             cv2.rectangle(image, (global_sx1, global_sy1), (global_sx2, global_sy2), (0, 255, 0), 2)

    # if windshield_boxes:
    #     for x1, y1, x2, y2 in windshield_boxes:
    #         x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    #         cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
    #     output_path = os.path.join(output_folder, image_filename)
    #     cv2.imwrite(output_path, image)
    #     print(f"Saved windshield detections to {output_path}")
    #     break
        
    # output_path = os.path.join(output_folder, img_filename)
    # cv2.imwrite(output_path, image)

    # break

# display accuracy result
