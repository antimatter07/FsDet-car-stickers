# Few-Shot Object Detection (FsDet)

This repository contains the implementation of FsDet for Few-Shot Car Sticker Detection experiments. The code base was built upon the official few-shot object detection implementation of the ICML 2020 paper
[Frustratingly Simple Few-Shot Object Detection](https://arxiv.org/abs/2003.06957).
![FsDet Architecture](../assets/fewshot_training_diagram.png)




## Installation

**Requirements**

* Linux with Python >= 3.6
* [PyTorch](https://pytorch.org/get-started/locally/) >= 1.4
* [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation
* CUDA 9.2, 10.0, 10.1, 10.2, 11.0
* GCC >= 4.9

**Build FsDet**
* Create a virtual environment.
```angular2html
python3 -m venv fsdet
source fsdet/bin/activate
```
You can also use `conda` to create a new environment.
```angular2html
conda create --name fsdet
conda activate fsdet
```
* Install PyTorch. Please choose the Pytorch and CUDA version matches your machine's specs. Also ensure PyTorch version matches the prebuilt Detectron2 version (next step). Example for PyTorch v1.6.0:
```angular2html
pip install torch==1.6.0 torchvision==0.7.0
```
Currently, the codebase is compatible with [Detectron2 v0.2.1](https://github.com/facebookresearch/detectron2/releases/tag/v0.2.1), [Detectron2 v0.3](https://github.com/facebookresearch/detectron2/releases/tag/v0.3), and [Detectron2 v0.4](https://github.com/facebookresearch/detectron2/releases/tag/v0.4). Tags correspond to the exact version of Detectron2 that is supported. To checkout the right tag (example for Detectron2 v0.3):
```bash
git checkout v0.3
```

To install depedencies (example for PyTorch v1.6.0, CUDA v10.2, Detectron2 v0.3):
* Install Detectron2 v0.3
```angular2html
python3 -m pip install detectron2==0.3 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.6/index.html
```
* Install other requirements.
```angular2html
python3 -m pip install -r requirements.txt
```





## Usage Guide

### Training & Evaluation in Command Line

To train a model with settings encoded into a YAML file, run
```angular2html
python3 -m tools.train_net --num-gpus 1 \
        --config-file configs/stickers-detection/stickers_only_31shot.yaml
```

To evaluate the trained models, run
```angular2html
python3 -m tools.test_net --num-gpus 1 \
        --config-file configs/stickers-detection/stickers_only_31shot.yaml \
        --eval-only
```

To evaluate on all saved weights saved with `CHECKPOINT_PERIOD`,

```
python3 -m tools.test_net --num-gpus 1 --config-file configs/stickers-detection/stickers_only_31shot.yaml --eval-all
```

To initialize weights randomly befor few-shot fine tuning,
```
python3 -m tools.ckpt_surgery --src1 checkpoints/base/coco/model_final.pth --method randinit --save-dir checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_stickersnovel --stickers
```
The argument `src1` is the path to base trained weights you would like to use, please make sure to create a custom script for correctly mapping classes in `tools.ckpt_surgery.py` for your use case. In our experiments, base trained weights are weights base trained on COCO by original FsDet researchers found [here](http://dl.yf.io/fs-det/models/coco/base_model/).

## Windshield-to-Sticker Detection (WiSDet)
To run WiSDet pipeline or inference on full-image similar to built-in evaluation .py file,
```
python test_wscs_args.py \
    --mode ws-then-cs \
    --ws-config configs/windshield.yaml \
    --ws-weights weights/windshield.pth \
    --cs-config configs/sticker.yaml \
    --cs-weights weights/stickers.pth \
    --input-folder datasets/test_images \
    --output-folder results/wisdet_ws_cs/ \
    --ws-score-thresh 0.70 \
    --cs-score-thresh 0.05
```



# Command-Line Arguments

| Argument | Description |
|----------|-------------|
| `--mode` | `ws-then-cs` (WiSDet pipeline) or `cs-only` (full-image detection) |
| `--ws-config` | YAML config for windshield model |
| `--ws-weights` | `.pth` weights for windshield model |
| `--cs-config` | YAML config for sticker model |
| `--cs-weights` | `.pth` weights for sticker model |
| `--input-folder` | Folder containing test images |
| `--output-folder` | Where JSON & visualizations will be saved |
| `--dataset-name` | Detectron2 registry name (optional) |
| `--ws-score-thresh` | Score threshold for windshield |
| `--cs-score-thresh` | Score threshold for stickers |

