import os
import fsdet.data.meta_stickers
import fsdet.data.builtin
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
import cv2
from detectron2.structures import BoxMode

# --- CONFIGURATION ---
# stickers_ws_31shot_1280_test_tinyonly_top4
dataset_name = "stickers_ws_31shot_1280_test_tinyonly_top4"
output_folder = "gt_images_with_annotations/31shot_test_red"
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
        instance_mode=ColorMode.IMAGE  # no color augmentation
    )
    vis._default_font_size = 12
    vis.line_width = 5
    vis.alpha = 1.0

    # Draw each annotation in red
    for ann in data["annotations"]:
        bbox = ann["bbox"]
        if ann["bbox_mode"] != BoxMode.XYXY_ABS:
            bbox = BoxMode.convert(bbox, ann["bbox_mode"], BoxMode.XYXY_ABS)

        category_id = ann.get("category_id", 0)
        category_name = metadata.thing_classes[category_id] if "thing_classes" in metadata.as_dict() else str(category_id)

        vis.draw_box(bbox, edge_color=(1.0, 0.0, 0.0))  # pure red in RGB
        vis.draw_text(
            category_name,
            (bbox[0], bbox[1] - 5),
            color="red",
            font_size=vis._default_font_size,
        )

    output_image = vis.output.get_image()[:, :, ::-1]
    save_path = os.path.join(output_folder, f"{os.path.basename(img_path)}_GT.jpg")
    cv2.imwrite(save_path, output_image)

    print(f"[{i+1}/{len(dataset_dicts)}] Saved GT visualization to: {save_path}")

print(f"\n Done! All GT visualizations saved to: {output_folder}")
