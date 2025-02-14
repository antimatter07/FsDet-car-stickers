# WIP - Viene
from fsdet.engine import 
from fsdet.config import get_cfg

# config separate FSDet for windshield and stickers
cfg_windshield = get_cfg()
cfg_windshield.merge_from_file("configs/COCO-detection/faster_rcnn_R_101_FPN_ft_novel_5shot.yaml") 
windshield_fsdet = 


# load image/s

# for each image

    # detect windshield and store map of its coordinates

    # detect stickers within coordinates

# display result