# YOLO + WiSDet Guided Pipeline

This an implementation of the two stage guided detection pipeline WiSDet(Windshield-to-Sticker Detection) with YOLOv8 in a modular pipeline. This implementation can accomodate different object detection models as long as the guidelines for the pipeline are followed.

The pipeline works as follows:

1. A **guide detector** finds the windshield on the full image.  
2. A **sticker detector** runs on cropped windshields to detect tiny stickers.  
3. Sticker boxes from crops are **remapped** back to original image coordinates.  
4. Performance is evaluated with **COCO metrics** and optionally an **AP50 based learning rate scheduler**.

This can be run in two ways:

- Through a **Jupyter notebook** (`YOLO + WiSDet Notebook.ipynb`)  
- Through a **Python script** such as `Sample_Usage.py` (or your own `main.py`)

---

## Requirements

### Python

- Python 3.10 or newer  
- CUDA capable GPU is strongly recommended for training

### Key packages

- `torch`  
- `ultralytics`  
- `numpy`  
- `opencv-python`  
- `matplotlib`  
- `tqdm`  
- `pycocotools`  
- `pyyaml`  

Plus standard libraries such as `pathlib`, `json`, `time`, `uuid`, `shutil`, `glob`.

Example install:

```
pip install ultralytics pycocotools opencv-python pyyaml tqdm
```

---
## Project structure

Core modules:

- `BaseDetector.py`  
  Abstract interface for any detector that participates in the guided pipeline.  
  It defines:
  - `train_model(...)`
  - `predict_windshield(...)`
  - `predict_sticker(...)`  
  All detectors must follow the return formats described in the docstrings so the pipeline can consume their outputs.

- `YOLODetector.py`  
  YOLOv8 based implementation of `BaseDetector`. It wraps `ultralytics.YOLO` and provides:
  - Training utilities with optional AP50 based scheduling using `ReduceOnPlateauMAP50_WithDetectorNNClone`
  - `predict_windshield` on full images
  - `predict_sticker` on crops

- `WiSDet_Pipeline.py`  
  Contains:
  - `GuidedPipeline` which wires together a **guide** detector and a **sticker** detector, handles cropping, building the crop dataset, remapping, and simple visualization
  - `build_sticker_crop_dataset` to create a YOLO dataset of sticker crops for stage two training
  - `evaluate_fn`, a COCO based evaluation function that computes AP and AR from pipeline predictions

- `Scheduler.py`  
  Contains `ReduceOnPlateauMAP50_WithDetectorNNClone`, an external AP50 based learning rate scheduler that clones the current detector, evaluates AP50 via `evaluate_fn`, and reduces learning rate when performance stalls.

- `config.py`  
  Holds central dataset paths:
  - `DATA_YAML`: YOLO data YAML for the windshield dataset  
  - `COCO_SPLIT`: COCO style split used for training AP evaluation  
  - `COCO_SPLIT_TEST`: COCO style split for test evaluation

- `visualize_full_eval_function.py`  
  Contains `evaluate_fn_tags_visualize`, an evaluation function that evaluates the AP and recall of the predictions by viewpoint. The function is also able to perform visualizations.


- `Sample_Usage.py`  
  Concrete script that shows how to:
  - Load dataset paths from `config.py`
  - Instantiate two `YOLODetector` models (guide and sticker)
  - Build a `GuidedPipeline`
  - Run two stage training, prediction on the test split, and AP50 evaluation with `evaluate_fn`

- `Sample_Usage_Visual_Eval.py`  
  Concrete script that shows how to:
  - Utilize `evaluate_fn_tags_visualize` in visualizing and evaluating predictions.

Notebook:

- `YOLO + WiSDet Notebook.ipynb`  
  Jupyter notebook version that contains the same ideas in a more interactive flow, with inline visualizations and step by step sections.

---

## Notebook structure (short version)

Inside `YOLO + WiSDet Notebook.ipynb` you will find:

1. **Setup**  
   Imports, definition of `BaseDetector`, and common utilities.

2. **YOLOv8 detector wrapper**  
   Defines `YOLODetector` as a concrete `BaseDetector`.

3. **Guided pipeline**  
   Defines `GuidedPipeline` and helper methods for cropping, indexing crops, remapping predictions, and simple visualization.

4. **COCO evaluation**  
   Defines `evaluate_fn` to convert predictions into COCO format and compute AP and AR.

5. **Full COCO evaluation with Visualization**  
   Defines `evaluate_fn_tags_visualize` to convert predictions into COCO format and compute AP and AR by viewpoint with prediction visualizations.

6. **AP50 based scheduler**  
   Defines `ReduceOnPlateauMAP50_WithDetectorNNClone` and shows how to register it as Ultralytics callbacks.

7. **Example script cells**  
   Example cells that set dataset paths, instantiate detectors, build the pipeline, train both stages, run inference, and compute AP50.

---

## Dataset layout

The code uses two different dataset formats.

1. YOLO dataset for training

Described by DATA_YAML in config.py. Example:
``` 
path: /path/to/windshield_dataset
train: images/train
val: images/val
test: images/test
names:
  0: car-sticker
  1: windshield
```
YOLO folder layout:
``` 
/path/to/windshield_dataset/
  train/
    images/
    labels/
  val/
    images/
    labels/
  test/
    images/
    labels/
```

2. COCO style dataset for evaluation
Used by `evaluate_fn` and referenced by `COCO_SPLIT` and `COCO_SPLIT_TEST` in config.py.

Each COCO split directory should contain:
```
/path/to/split/
  _annotations.coco.json
  <image files>
```
Notes:

- COCO categories must correspond to your model classes.
- If COCO category ids differ from YOLO class indices, you can pass a class_map to evaluate_fn.

---
## How to run the code
You can either run the notebook or run the Python script.

**Option A: Jupyter notebook**

1. Open the notebook

- In Colab: upload the project files or mount Google Drive, then open YOLO + WiSDet Notebook.ipynb.
- Locally: use Jupyter Lab or VS Code to open the same notebook.

2. Install dependencies
- Run a setup cell that installs any missing packages, for example:
```
pip install ultralytics pycocotools opencv-python pyyaml tqdm
```

3. Set your paths
- Edit the notebook cells that define:
  - DATA_YAML (YOLO car sticker with windshields dataset)
  - COCO_SPLIT and COCO_SPLIT_TEST (for evaluation)

4. Run the pipeline
Typical flow in the notebook:
- Train the guide detector on windshields
- Use guide predictions to crop windshields and build the sticker crop dataset
- Train the sticker detector
- Run full two stage inference on the test split
- Call evaluate_fn to compute AP, AP50, and related metrics

5. Inspect visualizations
The notebook will show sample crops and sample images with ground truth and remapped predictions.

**Option B: Python script (Sample_Usage.py or main.py)**

If you prefer a pure Python workflow without the notebook:

1. Update config.py
Edit `config.py` so that:
```
DATA_YAML = "/path/to/your/windshield/data.yaml"
COCO_SPLIT = "/path/to/your/COCO/train"
COCO_SPLIT_TEST = "/path/to/your/COCO/test"
```
These are used by YOLODetector and the scheduler to find train and test splits for AP evaluation.

2. Check the sample script
`Sample_Usage.py` contains a `main()` function that:
- Creates two YOLODetector instances, one for the guide (windshield) and one for the sticker detector
- Builds a GuidedPipeline with:
  - detector=sticker_model
  - guide=windshield_model
  - thresholds and image size
- Calls pipeline.train(...) for the two stage training
- Calls pipeline.predict(...) on the test split
- Evaluates AP50 with evaluate_fn and prints the final AP50 value

3. Run the script
From the project root:
```
python Sample_Usage.py
```
The script will:
- Train the guide detector on the windshield dataset
- Build the sticker crop dataset from predicted windshields
- Train the sticker detector on the cropped patches
- Run the full two stage pipeline on the test split
- Compute AP50 with evaluate_fn and print the result

4. Tuning and extensions
In `Sample_Usage.py` you can tweak:
- pretrained weights for both detectors
- conf, iou, and input_size in the `GuidedPipeline` constructor
- epochs for guide and sticker inside `pipeline.train`
- `scheduled_epochs` if you want specific learning rate steps together with the AP50 scheduler