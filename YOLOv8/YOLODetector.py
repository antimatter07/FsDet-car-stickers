from ultralytics import YOLO
import torch
import yaml, os, cv2, uuid, shutil
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from shutil import copy2

from BaseDetector import BaseDetector

class YOLODetector(BaseDetector):
    """YOLOv8 based implementation of :class:`BaseDetector` for the Guided Pipeline.

    This class wraps an ``ultralytics.YOLO`` model and provides:

      * Training utilities that integrate with a Reduce-on-Plateau scheduler.
      * Windshield prediction on full images in the Guided Pipeline format.
      * Sticker prediction on cropped images.

    The class is intended to be used as the guide or sticker detector within a
    two stage Guided Pipeline.
    """

    def __init__(
        self,
        pretrained: str = "yolov8n.pt",
        lr0: float = 0.001,
        momentum: float = 0.95,
        weight_decay: float = 0.0005,
        freeze: int = 10,
        ReduceLRonPlateau: bool = False,
    ):
        """Initialize a YOLODetector instance.

        Args:
            pretrained: Path to a pretrained YOLOv8 checkpoint or a model name
                that ``ultralytics.YOLO`` can load, for example ``"yolov8n.pt"``.
            lr0: Initial learning rate used during training.
            momentum: Momentum parameter for the SGD optimizer.
            weight_decay: Weight decay (L2 regularization) used during training.
            freeze: Number of layers or blocks that YOLO should freeze during
                training. Passed to ``YOLO.train`` via the ``freeze`` argument.
            ReduceLRonPlateau: If ``True``, enable learning rate scheduling using
                the custom ``ReduceOnPlateauMAP50_WithDetectorNNClone`` callback.
        """
        self.pretrained = pretrained
        self.model = YOLO(pretrained)

        self.lr0 = lr0
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.freeze = freeze
        self.ReduceLRonPlateau = ReduceLRonPlateau

        print('pretrained:', pretrained)

    def get_split_dir(self, yaml_path, split: str = "train") -> str:
        """Get the image directory for a given split from a YOLO data YAML.

        This helper assumes the standard directory layout:

            <dataset_root>/
                data.yaml
                train/
                    images/
                    labels/
                val/
                    images/
                    labels/
                test/
                    images/
                    labels/

        Args:
            yaml_path: Path to a YOLO data YAML file.
            split: Dataset split key such as ``"train"``, ``"val"`` or ``"test"``.

        Returns:
            The path to the corresponding ``<split>/images`` directory as a
            string.
        """
        return str(Path(yaml_path).parent / split / "images")

    def _ensure_overrides_model(self):
        """Ensure that the underlying YOLO model has a valid ``overrides['model']``.

        Some ultralytics workflows expect ``model.overrides['model']`` to contain
        the model configuration path or checkpoint. This method sets that field
        if it is missing or ``None``, using either ``model.cfg`` or the
        ``pretrained`` path passed at initialization.
        """
        m = self.model
        if not hasattr(m, "overrides"):
            m.overrides = {}
        if "model" not in m.overrides or m.overrides["model"] is None:
            m.overrides["model"] = getattr(m, "cfg", None) or self.pretrained

    ########################################################
    def train_model(
        self,
        data_yaml,
        classes=None,
        epochs: int = 300,
        imgsz: int = 1200,
        seed: int = 30,
        AP_split: str = 'train',
        yolo_names=['car-sticker'],
        scheduled_epochs=None,
        conf: float = 0.5,
        iou: float = 0.5,
        **kwargs,
    ):
        """Train the YOLO model as a guide or windshield detector.

        This method wraps ``ultralytics.YOLO.train`` and optionally attaches a
        ``ReduceOnPlateauMAP50_WithDetectorNNClone`` callback to adjust the
        learning rate based on AP@50 on a COCO style evaluation.

        The intent is to train the first stage of the Guided Pipeline on
        windshield or guide bounding boxes.

        Args:
            data_yaml: Path to the YOLO data YAML that defines train and val
                splits and class names.
            classes: Optional list of class indices to train on. If ``None``,
                YOLO uses all classes defined in the YAML.
            epochs: Number of training epochs.
            imgsz: Input image size. Passed to YOLO as ``imgsz``.
            seed: Random seed for reproducibility.
            AP_split: Dataset split to use when evaluating AP within the
                scheduler, for example ``"train"`` or ``"val"``.
            yolo_names: List of class names used when computing AP metrics in
                the scheduler.
            scheduled_epochs: Optional list of epochs where the LR should be
                changed explicitly by the scheduler. If ``None``, no fixed
                schedule is applied.
            conf: Confidence threshold used during internal AP evaluation.
            iou: IoU threshold used during internal AP evaluation.
            **kwargs: Additional keyword arguments forwarded to
                ``YOLO.train``, such as augmentation options or logging flags.

        Returns:
            None. The underlying YOLO training call returns a ``Results``
            object, but this method does not forward it. Training history and
            test metrics are recorded inside the scheduler callback.
        """
        if scheduled_epochs is None:
            scheduled_epochs = []

        Global_History = []
        Test_History = []

        hyp_arr = {
            'lr0': self.lr0,
            'momentum': self.momentum,
            'weight_decay': self.weight_decay,
        }

        hyp = hyp_arr
        detector_factory = lambda: YOLODetector(pretrained=self.pretrained)     # Model copy

        plateau_cb = ReduceOnPlateauMAP50_WithDetectorNNClone(
            # ReduceOnPlateaurAP50
            use_plateau=self.ReduceLRonPlateau,                                  # turn off patience based logic
            factor=0.1,
            patience=10,                                                         # first plateau needs 100 bad epochs
            patience_after_first=10,                                             # after first LR drop, use 50
            cooldown=0,
            min_lr=0.000001, warmup_epochs=0, start_after_map=-1.0,             # patience counts only after AP50 > ...

            # MultiStepLR
            scheduled_epochs=scheduled_epochs,                                   # reduce exactly at these epochs
            scheduled_factors={},                                                # fallback to .factor if missing Epoch:Factor

            # For AP Generation
            owner=self,
            detector_factory=detector_factory,
            evaluate_fn=evaluate_fn,
            coco_split_dir=COCO_SPLIT,
            yolo_names=yolo_names,
            predict_kwargs=dict(
                data_yaml=DATA_YAML, classes=classes,
                imgsz=(800, 1200), AP_split=AP_split,
                conf=conf, iou=iou, verbose=False, save=False
            ),
            eval_every=10,
            clone_device="cuda:0",                                               # "cpu" or "cuda:0" if you prefer speed
            verbose=True,

            # Only for epoch testing, this does not affect training
            test_eval_start_epoch=999999999,
            test_eval_epochs=[],
            test_eval_every=99999,
            test_predict_kwargs=dict(                                            # only deltas from predict_kwargs are needed
                split="test",                                                    # ensure test split
                data_yaml=DATA_YAML
            ),
            test_coco_split_dir=COCO_SPLIT_TEST
        )

        # Register hooks
        self.model.add_callback("on_train_epoch_start", plateau_cb.on_train_epoch_start)
        self.model.add_callback("on_fit_epoch_end", plateau_cb)
        self.model.add_callback("on_fit_end", plateau_cb.on_fit_end)
        self.model.add_callback("on_train_end", plateau_cb.on_train_end)

        # Train
        results = self.model.train(
            data=data_yaml, classes=classes,
            epochs=epochs, imgsz=imgsz,
            conf=conf, iou=iou,
            optimizer="SGD", lr0=hyp['lr0'],
            momentum=hyp['momentum'],
            weight_decay=hyp['weight_decay'],
            seed=seed,
            batch=4,
            device=0,
            cos_lr=False,
            lrf=1.0,
            warmup_epochs=0,
            freeze=self.freeze,
            **kwargs
        )

        Global_History.append({'history': plateau_cb.history})
        Test_History.append({'Metrics:': plateau_cb.test_ap_history})

    ########################################################
    def predict_windshield(
        self,
        data_yaml,
        classes=None,
        imgsz=(800, 1200),
        split: str = 'test',
        conf: float = 0.5,
        iou: float = 0.5,
        **kwargs,
    ):
        """Run guide or windshield detection on full images.

        This method scans the image directory corresponding to the requested
        split in the given YOLO data YAML, runs YOLO inference, and returns the
        best windshield bounding box per image in the Guided Pipeline format.

        Args:
            data_yaml: Path to the YOLO data YAML file.
            classes: Optional list of class indices to detect. If ``None``, use
                all classes.
            imgsz: Image size (height, width) to use during inference.
            split: Dataset split key, for example ``"train"`` or ``"test"``.
            conf: Minimum confidence threshold for YOLO predictions.
            iou: IoU threshold used for internal non maximum suppression.
            **kwargs: Additional keyword arguments passed to
                ``YOLO.predict`` such as device, half precision, or visualization
                flags.

        Returns:
            list[dict]: A list with one entry per image, where each entry has
            the structure:

                {
                    "image_path": "<absolute or relative image path>",
                    "guides": [
                        {
                            "xyxy": [x1, y1, x2, y2],
                            "conf": float_confidence,
                            "cls": class_id
                        }
                    ]
                }

            If an image has no detections, the ``"guides"`` list is empty.
        """
        data_dir = self.get_split_dir(data_yaml, split)
        print('data_dir: ', data_dir)

        stream = self.model.predict(
            source=data_dir,
            conf=conf, iou=iou,
            classes=classes,
            imgsz=imgsz,
            stream=True,
            save=True,
            **kwargs
        )

        predicted_ws = []

        for r in stream:
            img_path = str(r.path)

            if r.boxes is None or len(r.boxes) == 0:
                predicted_ws.append({"image_path": img_path, "guides": []})
                continue

            # tensors -> numpy
            xyxy = r.boxes.xyxy.cpu().numpy()              # (N,4) [x1,y1,x2,y2]
            score = r.boxes.conf.cpu().numpy()             # (N,)
            cls = r.boxes.cls.cpu().numpy() if r.boxes.cls is not None else None

            # sort by confidence (desc)
            order = score.argsort()[::-1]
            xyxy = xyxy[order]
            score = score[order]
            if cls is not None:
                cls = cls[order]

            # index of best detection
            best_idx = int(score.argmax())

            b = xyxy[best_idx].tolist()
            g = {"xyxy": b, "conf": float(score[best_idx])}
            if cls is not None:
                g["cls"] = int(cls[best_idx])

            print(f"{img_path}: keeping best box -> conf={score[best_idx]:.3f}")

            predicted_ws.append({"image_path": img_path, "guides": [g]})

        return predicted_ws

    def predict_sticker(
        self,
        data_yaml,
        classes=None,
        imgsz=(800, 1200),
        split: str = 'train',
        conf: float = 0.5,
        iou: float = 0.5,
        save: bool = False,
        **kwargs,
    ):
        """Run sticker detection on cropped or full images.

        This method operates on all images in the specified split of the data
        YAML, runs YOLO inference, and returns all bounding box predictions for
        each image.

        It is intended for the second stage of the Guided Pipeline where the
        model predicts stickers inside previously cropped windshield regions or
        inside full images.

        Args:
            data_yaml: Path to the YOLO data YAML file.
            classes: Optional list of class indices to detect. If ``None``, use
                all classes.
            imgsz: Image size (height, width) to use during inference.
            split: Dataset split key, for example ``"train"`` or ``"test"``.
            conf: Minimum confidence threshold for YOLO predictions.
            iou: IoU threshold used for internal non maximum suppression.
            save: If ``True``, YOLO saves visualization outputs in its standard
                ``runs/detect`` directory.
            **kwargs: Additional keyword arguments forwarded to
                ``YOLO.predict``.

        Returns:
            list[dict]: A list with one entry per image, where each entry has
            the structure:

                {
                    "crop_path": "<image path>",
                    "boxes": [
                        {
                            "xyxy": [x1, y1, x2, y2],
                            "conf": float_confidence,
                            "cls": class_id
                        },
                        ...
                    ]
                }

            If an image has no detections, ``"boxes"`` is an empty list.
        """
        data_dir = self.get_split_dir(data_yaml, split)
        print('data_dir: ', data_dir)

        print("Prediction Start...\n\n")

        results = self.model.predict(
            source=data_dir,
            conf=conf, iou=iou,
            imgsz=imgsz,
            classes=classes,
            stream=True,
            save=save
        )

        preds = []
        for i, r in enumerate(results, 1):
            path = str(r.path)
            H, W = r.orig_shape  # (H, W)
            ms = float(r.speed['inference'])

            if r.boxes is None or len(r.boxes) == 0:
                preds.append({"crop_path": path, "boxes": []})
                continue

            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clses = r.boxes.cls.cpu().numpy() if r.boxes.cls is not None else None

            # sort by confidence (desc), keep all
            order = confs.argsort()[::-1]
            xyxy = xyxy[order]
            confs = confs[order]
            if clses is not None:
                clses = clses[order]

            boxes = []
            for j in range(len(confs)):
                b = {
                    "xyxy": xyxy[j].tolist(),
                    "conf": float(confs[j])
                }
                if clses is not None:
                    b["cls"] = int(clses[j])
                boxes.append(b)

            preds.append({"crop_path": path, "boxes": boxes})

        return preds