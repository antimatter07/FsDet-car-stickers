"""Example script for training and evaluating the GuidedPipeline.

This example shows how to:

1. Set up dataset paths for YOLO and COCO.
2. Instantiate two YOLODetector instances:
   * windshield_model as the guide or windshield detector.
   * sticker_model as the sticker detector.
3. Build a GuidedPipeline that uses:
   * guide to detect windshields on full images.
   * detector to detect stickers inside cropped windshields.
4. Train the two stage pipeline:
   * Stage 1: Train the guide on windshields.
   * Stage 2: Use predicted windshields to build a crop dataset and train the sticker detector.
5. Run inference on the test split:
   * Predict windshields.
   * Crop them.
   * Predict stickers on crops.
   * Remap sticker predictions back to original image coordinates.
6. Evaluate the final sticker detections with COCO metrics using evaluate_fn.

The GuidedPipeline.predict method returns a dict that maps original image paths
to a list of remapped sticker detections:

    {
        "<original/image.jpg>": [
            {
                "xyxy": [x1, y1, x2, y2],
                "conf": float_confidence,
                "cls": int_class_id
            },
            ...
        ],
        ...
    }

This format is accepted directly by evaluate_fn as the predictions argument.

To run the full pipeline:

1. Adjust the paths in config.py so that DATA_YAML, COCO_SPLIT, and
   COCO_SPLIT_TEST match your environment.
2. Install the required dependencies (ultralytics, pycocotools, opencv python,
   matplotlib, numpy).
3. Run this module as a script. Training, prediction, and evaluation will
   execute in order.
"""

from config import DATA_YAML, COCO_SPLIT_TEST
from YOLODetector import YOLODetector
from WiSDet_Pipeline import GuidedPipeline, evaluate_fn


def main() -> None:
    # Create guide (windshield) and detector (sticker) models
    windshield_model = YOLODetector(pretrained="yolov8n.pt")
    sticker_model = YOLODetector(pretrained="yolov8n.pt")

    # Build the guided pipeline
    pipeline = GuidedPipeline(
        detector=sticker_model,
        guide=windshield_model,
        coco_split_dir=COCO_SPLIT_TEST,
        conf=[0.05, 0.7],     # [sticker_conf, windshield_conf]
        iou=[0.5, 0.7],       # [sticker_iou, windshield_iou]
        input_size=(800, 1200),
        seed=30,
    )

    # Two stage training:
    #   1) Train windshield detector, crop predictions, build sticker dataset.
    #   2) Train sticker detector on cropped windshield patches.
    pipeline.train(
        data_yaml=DATA_YAML,
        epochs=[100, 100],     # [sticker_epochs, windshield_epochs]
        scheduled_epochs=[],   # can be used together with ReduceOnPlateau scheduler
    )

    # Full two stage inference on the test split:
    #   1) Predict windshields.
    #   2) Crop windshields.
    #   3) Build a temporary sticker crop dataset.
    #   4) Predict stickers on the crops.
    #   5) Remap sticker predictions to original image coordinates.
    preds = pipeline.predict(
        data_yaml=DATA_YAML,
    )

    # COCO evaluation of remapped sticker predictions on the test split.
    # preds is a dict that maps original image paths to detection lists,
    # which matches the expected input format of evaluate_fn.
    ap50 = evaluate_fn(
        coco_split_dir=COCO_SPLIT_TEST,
        predictions=preds,
        yolo_names=["car-sticker"],
        target_size=(1280, 800),
    )
    print(f"Final AP50 on COCO test split: {ap50:.6f}")


if __name__ == "__main__":
    main()
