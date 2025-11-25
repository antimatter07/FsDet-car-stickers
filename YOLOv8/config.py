"""Configuration for dataset and COCO split paths.

Update these constants to match your local directory layout before running
training or evaluation scripts.

They are used by:
  * YOLODetector.train_model for AP evaluation through the scheduler.
  * Example usage scripts that construct and run the GuidedPipeline.
"""

# Path to the YOLO data YAML that describes the windshield dataset.
DATA_YAML = "/content/drive/MyDrive/Dataset/windshield/31shot_ws/data.yaml"

# Path to the COCO style split that contains the train annotations and images.
COCO_SPLIT = "/content/drive/MyDrive/Dataset_COCO/windshields/31shot_COCO/train"

# Path to the COCO style split that contains the test annotations and images.
COCO_SPLIT_TEST = "/content/drive/MyDrive/Dataset_COCO/windshields/31shot_COCO/test"

# Optional

# Path to COCO style dataset with left/right/near/far tags
DATA_TAGS='/content/drive/MyDrive/Dataset/Dataset with tags/All_in_test/test/_annotations.coco.json'


