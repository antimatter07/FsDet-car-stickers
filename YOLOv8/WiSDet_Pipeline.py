from tqdm import tqdm
import glob
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json, time
import matplotlib as plt
import yaml, os, cv2, uuid, shutil

from BaseDetector import BaseDetector

class GuidedPipeline:
    """Two stage guided detection pipeline for windshield and sticker detection.

    This pipeline coordinates two detectors that implement :class:`BaseDetector`:

      * ``guide`` - a guide or windshield detector that runs on full images and
        returns guide bounding boxes in the standard Guided Pipeline format:
        ``{"image_path": str, "guides": [{"xyxy": [...], "conf": float, "cls": int}, ...]}``.
      * ``detector`` - a sticker detector that runs on cropped windshield
        regions and returns sticker predictions in the standard format used by
        :meth:`BaseDetector.predict_sticker`.

    The typical flow is:

      1. Train the guide detector on full images for the windshield class.
      2. Use the guide detector to predict windshields and crop those regions.
      3. Build a YOLO dataset of sticker crops.
      4. Train the sticker detector on that crop dataset.
      5. At inference time, repeat steps 2 and 3, then run the sticker detector
         and remap crop coordinates back to the original image frame.

    The pipeline is designed to be compatible with :class:`YOLODetector` for
    both the guide and sticker stages.
    """

    def __init__(
        self,
        detector: BaseDetector,
        guide: BaseDetector,
        coco_split_dir,
        conf=[0.5, 0.5],
        iou=[0.5, 0.5],
        input_size=(960, 544),
        seed=30,
    ):
        """Initialize the GuidedPipeline.

        Args:
            detector: Sticker detector used in the second stage. Must implement
                :class:`BaseDetector` and return sticker predictions in the
                format used by :meth:`BaseDetector.predict_sticker`.
            guide: Guide or windshield detector used in the first stage. Must
                implement :class:`BaseDetector` and return windshields in the
                format used by :meth:`BaseDetector.predict_windshield`.
            coco_split_dir: Path to the COCO style split directory for this
                dataset. This is typically used by external evaluation utilities
                such as :func:`evaluate_fn`.
            conf: Two element list of confidence thresholds
                ``[sticker_conf, windshield_conf]`` used during prediction.
            iou: Two element list of IoU thresholds
                ``[sticker_iou, windshield_iou]`` used during prediction.
            input_size: Input resolution (width, height) that can be used by
                detectors. This value is stored but not enforced by the
                pipeline itself.
            seed: Seed for reproducibility. Forwarded to detector training
                calls where supported.
        """
        self.detector = detector
        self.guide = guide
        self.coco_split_dir = coco_split_dir
        self.conf = conf
        self.iou = iou
        self.input_size = input_size
        self.seed = seed

    ############################
    # Crop dataset utilities
    ############################

    def delete_folder_if_exists(self, folder_path):
        """Delete a folder and all its contents if it exists.

        Args:
            folder_path: Path to the folder to delete. This can be a string or
                a :class:`pathlib.Path` object.

        Side effects:
            Logs a message and removes the directory recursively if it exists.
        """
        path = Path(folder_path)
        if path.exists() and path.is_dir():
            print(f"Deleting existing folder: {path}")
            shutil.rmtree(path)

    def _yolo_line_to_xyxy(self, line: str, W: int, H: int):
        """Convert a YOLO label line to pixel coordinates in xyxy format.

        YOLO label lines have the form:

            cls cx cy w h

        where all coordinates are normalized relative to image width and height.

        Args:
            line: A single label line from a YOLO ``.txt`` file.
            W: Image width in pixels.
            H: Image height in pixels.

        Returns:
            Tuple ``(cls, x1, y1, x2, y2)`` where ``cls`` is the integer class
            id and ``x1, y1, x2, y2`` are the bounding box coordinates in pixels.
        """
        # "cls cx cy w h" (normalized) -> pixel xyxy
        c, cx, cy, w, h = map(float, line.split())
        bw, bh = w * W, h * H
        x1 = (cx * W) - bw / 2
        y1 = (cy * H) - bh / 2
        x2 = x1 + bw
        y2 = y1 + bh
        return int(c), x1, y1, x2, y2

    def _xyxy_to_yolo_line(self, x1, y1, x2, y2, W, H, cls=0):
        """Convert a pixel xyxy bounding box to a YOLO label line.

        Args:
            x1: Left coordinate of the bounding box in pixels.
            y1: Top coordinate of the bounding box in pixels.
            x2: Right coordinate of the bounding box in pixels.
            y2: Bottom coordinate of the bounding box in pixels.
            W: Image width in pixels.
            H: Image height in pixels.
            cls: Integer class id to write.

        Returns:
            A YOLO label line as a string in the format:

                "cls cx cy w h\\n"

            where ``cx, cy, w, h`` are normalized to the range [0, 1]. If the
            box is degenerate or has non positive width or height, returns
            ``None``.
        """
        # pixel xyxy -> "cls cx cy w h" normalized to W,H
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        if bw <= 0 or bh <= 0:
            return None
        cx = (x1 + x2) / 2.0 / W
        cy = (y1 + y2) / 2.0 / H
        w = bw / W
        h = bh / H
        # clip to [0,1] just in case
        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        w = min(max(w, 0.0), 1.0)
        h = min(max(h, 0.0), 1.0)
        if w <= 0 or h <= 0:
            return None
        return f"{int(cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"

    def _label_path_for_image(self, img_path: Path) -> Path:
        """Return the YOLO label path corresponding to an image path.

        This helper expects the standard YOLO layout where labels are stored
        in a sibling ``labels`` directory:

            .../<split>/images/<name>.jpg
            .../<split>/labels/<name>.txt

        Args:
            img_path: Path to an image in the ``images`` directory.

        Returns:
            Path to the corresponding label file in the ``labels`` directory.
        """
        return img_path.parent.parent / "labels" / (img_path.stem + ".txt")

    def _resolve_split_dir(self, data_yaml_path: Path, split_key: str) -> Path | None:
        """Resolve the directory path for a dataset split from a data YAML.

        This method reads the given data YAML and looks up the value under the
        specified split key (for example ``"train"``, ``"val"``, or ``"test"``).
        If the path in the YAML is relative, it is resolved relative to the
        directory containing the YAML file.

        Args:
            data_yaml_path: Path to the data YAML file.
            split_key: Key of the split to resolve, for example ``"test"``.

        Returns:
            A :class:`Path` to the split directory, or ``None`` if the split
            key is missing or empty.
        """
        y = yaml.safe_load(data_yaml_path.read_text())
        rel = y.get(split_key)
        if not rel:
            return None
        p = Path(rel)
        return p if p.is_absolute() else (data_yaml_path.parent / p)

    def build_sticker_crop_dataset(
        self,
        cropped_windshields,
        data_yaml,
        class_needed: str = 'car-sticker',
        out_root: str = "/content/sticker_crops",
    ):
        """Build a YOLO sticker crop dataset from cropped windshield regions.

        This method constructs a new YOLO dataset rooted at ``out_root`` with
        the following structure:

            out_root/
                train/
                    images/
                    labels/
                test/
                    images/
                    labels/
                data.yaml

        The train split is populated from the provided ``cropped_windshields``
        list, while the test split is copied from the original dataset defined
        in ``data_yaml`` (if a test split exists there).

        Bounding boxes from the original labels are intersected with each crop
        and remapped into crop coordinates, then written in YOLO format.

        Args:
            cropped_windshields: Iterable of crop records. Each record is
                expected to be a dictionary with keys:

                  * ``"crop"``: Crop image as a NumPy array (H, W, C).
                  * ``"box"``: Tuple ``(x1, y1, x2, y2)`` in original image
                    pixel coordinates describing the crop region.
                  * ``"image_path"``: Path to the original source image.

            data_yaml: Path to the original YOLO data YAML file. Class names
                and the original test split are read from this file.
            class_needed: Logical class name used to identify sticker classes,
                for example ``"car-sticker"``. The name is normalized to
                lowercase and underscores before matching against the YAML
                names.
            out_root: Root directory where the new crop dataset should be
                created. Existing contents at this path are deleted.

        Returns:
            str: Path to the newly created ``data.yaml`` file inside
            ``out_root``.

        Raises:
            ValueError: If ``cropped_windshields`` is empty.
        """
        import re  # local import is fine here

        def _norm(s: str) -> str:
            # normalize "Car-Sticker", "car sticker", etc. -> "car_sticker"
            return re.sub(r'[^a-z0-9]+', '_', s.strip().lower())

        self.delete_folder_if_exists(out_root)

        out_root = Path(out_root)
        (out_root / "train/images").mkdir(parents=True, exist_ok=True)
        (out_root / "train/labels").mkdir(parents=True, exist_ok=True)
        (out_root / "test/images").mkdir(parents=True, exist_ok=True)
        (out_root / "test/labels").mkdir(parents=True, exist_ok=True)

        if not cropped_windshields:
            raise ValueError("No cropped windshields provided.")

        data_yaml_path = Path(data_yaml)
        y = yaml.safe_load(data_yaml_path.read_text())

        # preserve class list/order
        names_raw = y.get("names")
        if isinstance(names_raw, dict):
            names = [names_raw[k] for k in sorted(map(int, names_raw.keys()))]
            write_as_dict = True
            id2name = {int(k): v for k, v in names_raw.items()}
        else:
            names = list(names_raw) if names_raw is not None else []
            write_as_dict = False
            id2name = {i: n for i, n in enumerate(names)}
        nc = len(names)

        # figure out which original class IDs correspond to "car_sticker"
        wanted_names = {class_needed}
        wanted_norm = {_norm(w) for w in wanted_names}
        sticker_ids = {i for i, n in id2name.items() if _norm(str(n)) in wanted_norm}
        if not sticker_ids:
            # fallback if names do not exist in YAML
            sticker_ids = {0}

        # original test dir (copy as-is)
        src_test_dir = self._resolve_split_dir(data_yaml_path, "test")

        # cache original image sizes
        shape_cache = {}
        EPS = 1e-6
        MIN_PX = 2

        # filename bookkeeping to preserve names and handle multiple crops per image
        # key: absolute original image path -> number of crops already written
        crop_counts = {}

        # write crops for TRAIN folder only
        for it in cropped_windshields:
            crop_img, (x1c, y1c, x2c, y2c), orig_img_path = it["crop"], it["box"], Path(it["image_path"])

            # base stem + extension from original
            base_stem = orig_img_path.stem
            ext = orig_img_path.suffix if orig_img_path.suffix else ".jpg"

            # pick filename: first crop keeps exact name, subsequent get suffix _c2, _c3, ...
            n = crop_counts.get(orig_img_path, 0) + 1
            crop_counts[orig_img_path] = n
            if n == 1:
                img_name = f"{base_stem}{ext}"
            else:
                img_name = f"{base_stem}_c{n}{ext}"

            img_out = out_root / "train/images" / img_name
            lbl_out = out_root / "train/labels" / (Path(img_name).stem + ".txt")

            # ensure uint8 3ch
            arr = crop_img
            if isinstance(arr, np.ndarray) and arr.dtype != np.uint8:
                arr = arr.astype(np.uint8)
            if arr.ndim == 2:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(str(img_out), arr)

            # map GT labels that intersect crop
            orig_lbl_path = self._label_path_for_image(orig_img_path)
            crop_W = max(EPS, (x2c - x1c))
            crop_H = max(EPS, (y2c - y1c))
            lines_out = []

            if orig_lbl_path.exists():
                if orig_img_path not in shape_cache:
                    im = cv2.imread(str(orig_img_path))
                    shape_cache[orig_img_path] = None if im is None else (im.shape[1], im.shape[0])  # (W,H)
                shape = shape_cache.get(orig_img_path)
                if shape is not None:
                    W, H = shape
                    with open(orig_lbl_path, "r") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            cls, bx1, by1, bx2, by2 = self._yolo_line_to_xyxy(line, W, H)

                            # intersect with crop (original coords)
                            ix1 = max(bx1, x1c)
                            iy1 = max(by1, y1c)
                            ix2 = min(bx2, x2c)
                            iy2 = min(by2, y2c)
                            if ix2 - ix1 < MIN_PX or iy2 - iy1 < MIN_PX:
                                continue  # no overlap / too tiny

                            # map to crop-local
                            cx1 = ix1 - x1c
                            cy1 = iy1 - y1c
                            cx2 = ix2 - x1c
                            cy2 = iy2 - y1c

                            yline = self._xyxy_to_yolo_line(
                                cx1, cy1, cx2, cy2, crop_W, crop_H, cls=int(cls)
                            )
                            if yline:
                                lines_out.append(yline)

            with open(lbl_out, "w") as f:
                for l in lines_out:
                    f.write(l)

        # copy ORIGINAL TEST split as-is
        if src_test_dir and (src_test_dir / "images").exists():
            for img in (src_test_dir / "images").glob("*.*"):
                if img.suffix.lower().lstrip(".") in {
                    'bmp', 'mpo', 'jpg', 'pfm', 'tif', 'tiff',
                    'png', 'webp', 'jpeg', 'dng', 'heic'
                }:
                    dst_img = out_root / "test/images" / img.name
                    dst_lbl = out_root / "test/labels" / (img.stem + ".txt")
                    shutil.copy2(img, dst_img)
                    lbl = src_test_dir / "labels" / (img.stem + ".txt")
                    if lbl.exists():
                        shutil.copy2(lbl, dst_lbl)

        # YAML: train and val both use train/, test uses test/ (if any)
        yaml_lines = [
            f"path: {out_root}",
            "train: train/images",
            "val: train/images",
        ]
        if (out_root / "test/images").exists() and any((out_root / "test/images").iterdir()):
            yaml_lines.append("test: test/images")
        yaml_lines += [f"nc: {nc}", "names:"]
        if write_as_dict:
            for idx, name in enumerate(names):
                yaml_lines.append(f"  {idx}: {name}")
        else:
            yaml_lines.append("  " + str(names))
        (out_root / "data.yaml").write_text("\n".join(yaml_lines) + "\n")

        print(
            'Built crop dataset with original filenames (subsequent crops use _c2/_c3 suffixes). '
            f"Root: {out_root}"
        )
        return str(out_root / "data.yaml")

    ############################
    # Core pipeline flow
    ############################

    def crop_windshields(self, predicted_ws):
        """Crop windshield regions from full images.

        This function consumes the guide predictions returned by
        :meth:`BaseDetector.predict_windshield` and extracts padded crops
        around each guide bounding box.

        Args:
            predicted_ws: List of guide prediction entries, where each entry
                has the form:

                    {
                        "image_path": "<path/to/image.jpg>",
                        "guides": [
                            {
                                "xyxy": [x1, y1, x2, y2],
                                "conf": float_confidence,
                                "cls": class_id
                            },
                            ...
                        ]
                    }

                If ``"guides"`` is empty, no crops are produced for that image.

        Returns:
            list[dict]: A list of crop records, each with the structure:

                {
                    "image_path": "<original/image/path.jpg>",
                    "crop": np.ndarray,              # cropped BGR image
                    "box": [x1c, y1c, x2c, y2c],     # crop box in original coords
                    "conf": float_confidence         # guide confidence
                }

            Entries whose crops are empty are skipped.
        """
        cropped_images = []
        pad = 5
        for item in tqdm(predicted_ws, desc="Cropping windshields"):
            img_path = item["image_path"]
            img = cv2.imread(img_path)
            if img is None:
                continue

            for g in item["guides"]:
                xy = g["xyxy"]
                conf = g.get("conf", 0.0)

                x1, y1, x2, y2 = map(int, xy[:4])

                x1c = max(0, x1 - pad)
                y1c = max(0, y1 - pad)
                x2c = min(img.shape[1], x2 + pad)
                y2c = min(img.shape[0], y2 + pad)
                crop = img[y1c:y2c, x1c:x2c]
                if crop.size == 0:
                    continue

                cropped_images.append({
                    "image_path": img_path,
                    "crop": crop,
                    "box": [x1c, y1c, x2c, y2c],
                    "conf": conf
                })
        return cropped_images

    def _make_crop_index(self, cropped_windshields, out_root="/content/sticker_crops"):
        """Build an index that maps crop file paths back to original images.

        This helper reconstructs the filenames that will be written by
        :meth:`build_sticker_crop_dataset` for the train split and builds a
        mapping from crop path to the original image path and crop box
        coordinates.

        It assumes the following behavior:

          * Crops are saved in ``out_root/train/images``.
          * The first crop for an original image keeps the base name
            ``<stem><ext>``.
          * Subsequent crops for the same original image receive suffixes
            ``_c2``, ``_c3``, and so on.

        Args:
            cropped_windshields: List of crop records produced by
                :meth:`crop_windshields`.
            out_root: Root directory where the sticker crop dataset is or will
                be created.

        Returns:
            dict: A mapping from crop image path to metadata:

                {
                    "<out_root>/train/images/<name>.jpg": {
                        "orig_path": "<original/image/path.jpg>",
                        "crop_box": [x1c, y1c, x2c, y2c]
                    },
                    ...
                }
        """
        out_root = Path(out_root)
        img_dir = out_root / "train" / "images"

        crop_counts = {}
        index = {}

        for it in cropped_windshields:
            orig_img_path = Path(it["image_path"])
            x1c, y1c, x2c, y2c = it["box"]
            base_stem = orig_img_path.stem
            ext = orig_img_path.suffix if orig_img_path.suffix else ".jpg"

            n = crop_counts.get(orig_img_path, 0) + 1
            crop_counts[orig_img_path] = n

            if n == 1:
                img_name = f"{base_stem}{ext}"
            else:
                img_name = f"{base_stem}_c{n}{ext}"

            crop_path = str(img_dir / img_name)

            index[crop_path] = {
                "orig_path": str(orig_img_path),
                "crop_box": [int(x1c), int(y1c), int(x2c), int(y2c)]
            }

        return index

    def remap_sticker_predictions(self, sticker_preds, crop_index):
        """Remap crop space sticker predictions back to original image space.

        This function takes sticker predictions produced by the second stage
        detector and moves each bounding box from crop coordinates into the
        coordinate frame of the original image, using the crop index built by
        :meth:`_make_crop_index`.

        Args:
            sticker_preds: Sticker predictions returned by
                :meth:`BaseDetector.predict_sticker`, typically with entries of
                the form:

                    {
                        "crop_path": "<path/to/crop.jpg>",
                        "boxes": [
                            {
                                "xyxy": [x1, y1, x2, y2],
                                "conf": float_confidence,
                                "cls": class_id
                            },
                            ...
                        ]
                    }

            crop_index: Mapping produced by :meth:`_make_crop_index` where each
                crop path is mapped to an original image path and crop box.

        Returns:
            dict: Mapping from original image path to a list of remapped boxes:

                {
                    "<original/image/path.jpg>": [
                        {
                            "xyxy": [X1, Y1, X2, Y2],   # original image coords
                            "conf": float_confidence,
                            "cls": class_id
                        },
                        ...
                    ],
                    ...
                }

            Entries whose crop paths are not found in ``crop_index`` are
            silently skipped.
        """
        remapped = {}

        for item in sticker_preds:
            crop_path = item["crop_path"]
            boxes = item["boxes"]

            if crop_path not in crop_index:
                # silently skip if crop is not in index (for example test images not from crops)
                continue

            info = crop_index[crop_path]
            orig_path = info["orig_path"]
            x1c, y1c, x2c, y2c = info["crop_box"]

            L = remapped.setdefault(orig_path, [])

            for b in boxes:
                x1, y1, x2, y2 = b["xyxy"]
                # shift from crop local to original image coords
                X1 = float(x1) + x1c
                Y1 = float(y1) + y1c
                X2 = float(x2) + x1c
                Y2 = float(y2) + y1c

                nb = {
                    "xyxy": [X1, Y1, X2, Y2],
                    "conf": b.get("conf", 0.0)
                }
                if "cls" in b:
                    nb["cls"] = b["cls"]
                L.append(nb)

        return remapped

    def train(
        self,
        data_yaml,
        epochs=[100, 100],
        scheduled_epochs=[],
        skip_windshield_train=False,
        skip_sticker_train=False,
    ):
        """Train the guide and sticker detectors in a two stage procedure.

        Stage 1 - guide or windshield detector:

          1. Train ``self.guide`` on the full dataset for the windshield class
             (class index 1 by convention).
          2. Predict windshields on the train split using the trained guide.
          3. Crop predicted windshields with :meth:`crop_windshields`.
          4. Build a sticker crop dataset and YAML using
             :meth:`build_sticker_crop_dataset`.

        Stage 2 - sticker detector:

          1. Train ``self.detector`` on the sticker crop dataset for the
             sticker class (class index 0 by convention).

        Either stage can be skipped if a pretrained model or prebuilt dataset
        already exists.

        Args:
            data_yaml: Path to the original YOLO data YAML file describing the
                full dataset.
            epochs: Two element list ``[sticker_epochs, windshield_epochs]``
                controlling the number of training epochs for each stage.
            scheduled_epochs: Reserved for integration with schedulers such as
                :class:`ReduceOnPlateauMAP50_WithDetectorNNClone`. Not used
                directly in this method but can be forwarded through
                ``**kwargs`` in custom implementations.
            skip_windshield_train: If ``True``, skip training the guide
                detector and assume ``data_yaml`` already points to a sticker
                crop dataset.
            skip_sticker_train: If ``True``, skip training the sticker
                detector.

        Returns:
            None. Training side effects are handled by the detector
            implementations.
        """
        if not skip_windshield_train:
            print("Training Windshield Detector...")
            windshield_ap = self.guide.train_model(
                data_yaml=data_yaml,
                classes=[1],
                epochs=epochs[1],
                conf=self.conf[1], iou=self.iou[1],
                seed=self.seed,
                yolo_names=['windshield']
            )

            print("Predicting Windshields for training...\n\n")
            predicted_ws = self.guide.predict_windshield(
                data_yaml=data_yaml,
                conf=self.conf[1], iou=self.iou[1],
                classes=[1],
                split='train'
            )

            print("Cropping windshields for sticker training...")
            cropped_windshields = self.crop_windshields(predicted_ws)

            # display 1 image
            if len(cropped_windshields) > 0:
                first_crop = cropped_windshields[0]["crop"]
                plt.figure(figsize=(6, 6))
                plt.imshow(cv2.cvtColor(first_crop, cv2.COLOR_BGR2RGB))
                plt.axis("off")
                plt.title("First cropped windshield")
                plt.show()
            else:
                print("No windshields were cropped.")

            print("\n\nBuilding Dataset...")
            sticker_yaml = self.build_sticker_crop_dataset(cropped_windshields, data_yaml)
            print('\n\nDataset Built Successfully.\n')

        else:
            sticker_yaml = data_yaml
            print("Skipping windshield training.")

        if not skip_sticker_train:
            print("Training Sticker Detector...")
            self.detector.train_model(
                data_yaml=sticker_yaml,
                classes=[0],
                epochs=epochs[0],
                conf=self.conf[0], iou=self.iou[0],
                seed=self.seed,
                yolo_names=['car-sticker']
            )

    def predict(self, data_yaml, **kwargs):
        """Run the full two stage pipeline at inference time.

        The prediction flow is:

          1. Use ``self.guide.predict_windshield`` on the test split of
             ``data_yaml`` to obtain guide boxes for each full image.
          2. Crop the predicted windshields with :meth:`crop_windshields`.
          3. Build a temporary sticker crop dataset and YAML via
             :meth:`build_sticker_crop_dataset`.
          4. Run ``self.detector.predict_sticker`` on the train split of the
             crop dataset (which contains all sticker crops).
          5. Remap crop space sticker predictions back to original image
             coordinates with :meth:`remap_sticker_predictions`.
          6. Visualize a small subset of results with
             :meth:`visualize_first5`.

        Args:
            data_yaml: Path to the original YOLO data YAML file describing the
                full dataset.
            **kwargs: Reserved for future options. Currently not used but can
                be wired to detectors if needed.

        Returns:
            dict: Remapped sticker predictions in the format produced by
            :meth:`remap_sticker_predictions`, that is:

                {
                    "<original/image/path.jpg>": [
                        {
                            "xyxy": [X1, Y1, X2, Y2],
                            "conf": float_confidence,
                            "cls": class_id
                        },
                        ...
                    ],
                    ...
                }
        """
        print("\n\nPredicting windshields...")
        predicted_ws = self.guide.predict_windshield(
            data_yaml=data_yaml,
            classes=[1],
            conf=self.conf[1], iou=self.iou[1],
            split='test'
        )

        print("\n\nCropping windshields...")
        windshield_results = self.crop_windshields(predicted_ws)

        # display 1 image
        if len(windshield_results) > 0:
            first_crop = windshield_results[0]["crop"]
            plt.figure(figsize=(6, 6))
            plt.imshow(cv2.cvtColor(first_crop, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.title("First cropped windshield")
            plt.show()

        else:
            print("No windshields were cropped.")

        print("\n\nBuilding Dataset...")
        sticker_yaml = self.build_sticker_crop_dataset(
            windshield_results,
            data_yaml
        )
        print('\n\nDataset Built Successfully.\n')

        print("\n\nPredicting Stickers...")
        # split = train because sticker_yaml has cropped images saved in train folder
        sticker_preds = self.detector.predict_sticker(
            data_yaml=sticker_yaml,
            conf=self.conf[0], iou=self.iou[0],
            classes=[0],
            split='train'
        )

        print("Sticker Detection Complete.")

        print("\n\nRemapping Sticker Predictions...")
        # build crop index using the SAME windshield_results and out_root used in build_sticker_crop_dataset
        crop_index = self._make_crop_index(windshield_results, out_root="/content/sticker_crops")

        # remap crop space predictions to original image coordinates
        remapped = self.remap_sticker_predictions(sticker_preds, crop_index)

        print("\n\nRemapping Complete...")

        # visualize first 5 originals with remapped predictions
        self.visualize_first5(remapped, max_show=5)

        return remapped

    def visualize_first5(self, remapped_preds, max_show=5, thickness=2):
        """Visualize a subset of remapped predictions against ground truth.

        For up to ``max_show`` original images, this function:

          * Loads the original image.
          * Draws ground truth boxes from the matching YOLO label file
            (class 0 only) in blue.
          * Draws remapped predictions in green, labeled with confidence and
            optional class id.
          * Shows the result with Matplotlib.

        Args:
            remapped_preds: Dictionary of remapped predictions returned by
                :meth:`predict`.
            max_show: Maximum number of images to visualize.
            thickness: Line thickness in pixels for the drawn rectangles.

        Returns:
            None.
        """
        shown = 0
        for orig_path, boxes in remapped_preds.items():
            if shown >= max_show:
                break

            img = cv2.imread(orig_path)
            if img is None:
                continue
            H, W = img.shape[:2]

            # draw GT from YOLO label file (blue)
            gt_count = 0
            p = Path(orig_path)
            lbl_path = p.parent.parent / "labels" / (p.stem + ".txt")  # .../<split>/labels/<name>.txt
            if lbl_path.exists():
                with open(lbl_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue
                        c, cx, cy, w, h = map(float, parts)   # cls cx cy w h  (normalized)

                        # filter: only show class 0 GT
                        if int(c) != 0:
                            continue

                        bw, bh = w * W, h * H
                        x1 = int(cx * W - bw / 2.0)
                        y1 = int(cy * H - bh / 2.0)
                        x2 = int(x1 + bw)
                        y2 = int(y1 + bh)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), thickness)  # blue = GT
                        cv2.putText(
                            img, f"GT c{int(c)}",
                            (x1, max(0, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 0, 0), 1, cv2.LINE_AA
                        )
                        gt_count += 1

            # draw predictions (green)
            pred_count = 0
            for b in boxes:
                x1, y1, x2, y2 = map(int, b["xyxy"])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness)  # green = Pred
                label = f"{b.get('conf', 0):.2f}"
                if "cls" in b:
                    label = f"c{b['cls']}:{label}"
                cv2.putText(
                    img, label,
                    (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 1, cv2.LINE_AA
                )
                pred_count += 1

            plt.figure(figsize=(8, 6))
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.title(f"{Path(orig_path).name} | preds: {pred_count} | gt: {gt_count}")
            plt.show()

            shown += 1
