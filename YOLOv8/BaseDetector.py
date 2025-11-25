from abc import ABC, abstractmethod

class BaseDetector(ABC):
    """Abstract base class for object detection models in the Guided Pipeline.

    The Guided Pipeline usually has two stages:
      - A guide or windshield detector that finds the region of interest.
      - A sticker detector that runs inside those regions.

    Subclasses can choose their own argument signatures for training and
    prediction (for example using a YOLO data YAML path), but they must
    follow the return formats described in the method docstrings so that
    the pipeline can consume their outputs.
    """

    @abstractmethod
    def train_model(self, data, **kwargs):
        """Train the detector model.

        Typical usage in the Guided Pipeline is to train the guide or
        windshield detector, but this method can also be used to train
        a sticker detector if needed.

        Args:
            data: Training data used to fit the model. The concrete
                implementation defines the exact type, for example:
                - Path to a YOLO data YAML file.
                - A dataset object or data loader.
                - A list of image paths or a dataset root directory.
            **kwargs: Additional keyword arguments for training. Implementations
                can support items such as:
                - Hyperparameters (epochs, batch size, learning rate).
                - Device selection.
                - Logging and checkpoint options.

        Returns:
            None. Implementations may optionally return internal training
            results, but the Guided Pipeline does not rely on a return value.
        """
        raise NotImplementedError

    @abstractmethod
    def predict_windshield(self, data, **kwargs):
        """Run guide or windshield detection on full images.

        This method must run inference on full images and return the best
        guide or windshield bounding box per image in a standard format.

        Args:
            data: Input data to run inference on. The concrete implementation
                defines the exact type, for example:
                - Path to a YOLO data YAML file, which is then used to find
                  an image directory for a split.
                - A list of image paths.
                - A dataset object or iterable of images.
            **kwargs: Extra keyword arguments to control inference such as
                confidence threshold, IoU threshold, image size, device,
                batch size, or whether to save visualizations.

        Returns:
            list[dict]: A list of dictionaries, one per image, with the
            following structure:

                [
                    {
                        "image_path": "<path/to/image.jpg>",
                        "guides": [
                            {
                                "xyxy": [x1, y1, x2, y2],
                                "conf": float_confidence,
                                "cls": class_id
                            }
                        ]
                    },
                    ...
                ]

            Notes:
                - "image_path" must be the path of the corresponding image.
                - "guides" is a list of predicted guide boxes. Implementations
                  may choose to keep only the best box (like YOLODetector) or
                  multiple boxes, but the value must always be a list.
                - If an image has no detections, "guides" must be an empty list.
        """
        raise NotImplementedError

    @abstractmethod
    def predict_sticker(self, data, **kwargs):
        """Run sticker detection on cropped or full images.

        This method is intended for the second stage of the Guided Pipeline.
        It typically receives cropped windshield images or a split of a
        dataset and returns all sticker detections for each image.

        Args:
            data: Input data for sticker inference. The concrete implementation
                defines the exact structure, for example:
                - Path to a YOLO data YAML file for a sticker crop dataset.
                - A list of crop image paths.
                - An iterable of images or crop records.
            **kwargs: Extra keyword arguments to control inference such as
                confidence threshold, IoU threshold, image size, device,
                batch size, or output saving options.

        Returns:
            list[dict]: A list of dictionaries, one per image or crop, with
            the following structure (as used by YOLODetector):

                [
                    {
                        "crop_path": "<path/to/crop_or_image.jpg>",
                        "boxes": [
                            {
                                "xyxy": [x1, y1, x2, y2],
                                "conf": float_confidence,
                                "cls": class_id
                            },
                            ...
                        ]
                    },
                    ...
                ]

            Notes:
                - "crop_path" should point to the image or crop used for
                  sticker prediction.
                - "boxes" is a list of all predicted sticker bounding boxes.
                - If an image has no detections, "boxes" must be an empty list.
        """
        raise NotImplementedError
