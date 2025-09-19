# import torch

# model_path = "checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_stickersnovelshot_tinyonly_top4classes/model_reset_surgery.pth" 

# # Load the model checkpoint
# checkpoint = torch.load(model_path, map_location="cpu")

# print("Checkpoint keys:")
# for k in checkpoint["model"]:
#     print(f"{k}: {checkpoint['model'][k].shape}")


# RUN TO SEE IF THERE ARE AVAILABLE GPUs
# import torch
# import GPUtil

# def get_best_gpu():
#     available_gpus = GPUtil.getAvailable(order='memory', limit=1)
#     return available_gpus[0] if available_gpus else None

# if torch.cuda.is_available():
#     best_gpu = get_best_gpu()
#     if best_gpu is not None:
#         device = torch.device(f"cuda:{best_gpu}")
#     else:
#         device = torch.device("cpu")
# else:
#     device = torch.device("cpu")

# print("Selected device:", device)


# RUN TO CHECK MODEL FEATURES
from fsdet.config import get_cfg
from fsdet.modeling import build_model

cfg = get_cfg()
cfg.merge_from_file("configs/stickers-detection/ws_then_cs.yaml")

model = build_model(cfg)
print("FPN features available:", model.backbone.output_shape().keys())