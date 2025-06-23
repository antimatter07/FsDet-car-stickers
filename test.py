import torch

model_path = "checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_stickersnovelshot_tinyonly_top4classes/model_reset_surgery.pth" 

# Load the model checkpoint
checkpoint = torch.load(model_path, map_location="cpu")

print("Checkpoint keys:")
for k in checkpoint["model"]:
    print(f"{k}: {checkpoint['model'][k].shape}")