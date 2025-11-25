"""Example script for evaluating a trained sticker detector with viewpoint based metrics.

This example shows how to:

1. Load a trained YOLODetector checkpoint that predicts stickers.
2. Run sticker detection directly on the test split defined in DATA_YAML.
3. Evaluate the raw YOLO predictions with a tag aware evaluation function that:
   * Computes AP metrics.
   * Computes recall broken down by viewpoint tags.
   * Optionally writes out visualizations to disk.

Unlike the two stage GuidedPipeline example:

* This script does not use a guide or windshield detector.
* It does not crop windshields or build a sticker crop dataset.
* It works directly on the full images defined by DATA_YAML and COCO_SPLIT_TEST.

Expected prediction format
--------------------------

The call to YOLODetector.predict_sticker returns a list of dictionaries:

    [
        {
            "crop_path": "<path/to/image_or_crop.jpg>",
            "boxes": [
                {
                    "xyxy": [x1, y1, x2, y2],
                    "conf": float_confidence,
                    "cls": int_class_id
                },
                ...
            ],
        },
        ...
    ]

This format is passed directly to evaluate_fn_tags_visualize as the
`predictions` argument. The evaluation function is responsible for aligning
these predictions with the COCO annotations and tag JSON.

To run this script:

1. Make sure `config.py` defines:
   * DATA_YAML: YOLO data yaml for this dataset.
   * COCO_SPLIT_TEST: path to the COCO style test split directory.
   * DATA_TAGS: path to a COCO style JSON with viewpoint or side tags.
2. Set `shot` and the checkpoint path to a valid trained YOLO model.
3. Ensure that the dependencies are installed:
   ultralytics, pycocotools, opencv python, matplotlib, numpy.
4. Run this module as a script to perform prediction and evaluation.
"""

from config import DATA_YAML, COCO_SPLIT_TEST, DATA_TAGS
from YOLODetector import YOLODetector
from visualize_full_eval_function import evaluate_fn_tags_visualize


def main() -> None:
    """Run sticker detector evaluation with viewpoint based metrics."""

    # For example, "10 shot" finetuned model
    shot = 10

    # Load a pretrained sticker detector checkpoint
    detector_model = YOLODetector(
        pretrained=(
            f"/content/drive/MyDrive/AA Guided SGD Results/"
            f"Original YOLO/Best Weights/nano/Best{shot}Shot.pt"
        )
    )

    # Run sticker prediction on the test split as defined in DATA_YAML.
    # The predict_sticker method returns a list of prediction records, each
    # with a "crop_path" and a list of "boxes".
    preds = detector_model.predict_sticker(
        data_yaml=DATA_YAML,
        classes=[0],              # sticker class index
        imgsz=(800, 1200),
        split="test",
        conf=0.05,
        iou=0.5,
    )

    # Evaluate with tag aware and visualization based evaluation.
    # This function can:
    #   * compute AP and recall per viewpoint or side tag
    #   * optionally save visualizations of detections
    metrics = evaluate_fn_tags_visualize(
        coco_split_dir=COCO_SPLIT_TEST,
        predictions=preds,
        yolo_names=["car-sticker"],
        target_size=(1200, 800),
        tags_coco_json=DATA_TAGS,

        # recall settings
        side_recall_mode="manual",
        recall_conf_thresh=0.05,

        # visualization params
        visualize_img_indices=None,   # None means let the function choose
        save_visuals=True,
        visual_conf_thresh=0.05,
        visual_box_alpha=0.45,
        visual_text_alpha=0.35,
    )

    print(metrics)


if __name__ == "__main__":
    main()
