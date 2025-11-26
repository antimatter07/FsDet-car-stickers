"""
draw_gt_annotations.py

Visualize ground truth annotations for a Detectron2/FsDet dataset.

This script iterates over all images in a dataset, draws the annotated bounding boxes
and labels using Detectron2's Visualizer, and saves the resulting images to an output folder.
"""

import os
import fsdet.data.meta_stickers
import fsdet.data.builtin
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
import cv2


if __name__ == "__main__":
    dataset_name = "crop_stickers_tinyonly_top4_train_10shot"
    output_folder = "gt_images_with_annotations/10shot_trainset_imageswithannots"
    os.makedirs(output_folder, exist_ok=True)
    
    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)
    
    print(f"> Loaded dataset '{dataset_name}' with {len(dataset_dicts)} images.")
    
    for i, data in enumerate(dataset_dicts):
        img_path = data["file_name"]
        image = cv2.imread(img_path)
        if image is None:
            print(f"[!] Could not read image: {img_path}")
            continue
    
    
        vis = Visualizer(
            image[:, :, ::-1],
            metadata=metadata,
            scale=1.0,
            instance_mode=None,  # disables color masking
        )
        
        vis._default_font_size = 12  # adjust for GT text labels
        vis.line_width = 5  # thicker lines
        vis.alpha = 1.0  # fully opaque
    
        vis_output = vis.draw_dataset_dict(data)
        output_image = vis_output.get_image()[:, :, ::-1]
    
        save_path = os.path.join(output_folder, f"{os.path.basename(img_path)}_GT.jpg")
        cv2.imwrite(save_path, output_image)
    
        print(f"[{i+1}/{len(dataset_dicts)}] Saved GT visualization to: {save_path}")
    
    print(f"\n Done! All GT visualizations saved to: {output_folder}")
