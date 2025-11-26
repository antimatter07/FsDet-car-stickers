# Windshield-to-Sticker Detection (WiSDet)
This repository contains the code used in the study **Few Shot Car Sticker Detection.** This study focuses on detecting tiny car stickers on images of cars with few annotated examples.

A two stage guided detection pipeline called **WiSDet (Windshield-to-Sticker Detection)**, is proposed for detecting car stickers on windshields. This pipeline first detects windshields then focuses sticker detection within the region.

![WiSDet Diagram](assets/windshield_guided_process.png)

The pipeline works as follows:

1. A **guide detector** finds the windshield on the full image.
2. The image is cropped to the windshield region.  
3. A **sticker detector** runs on the cropped images to detect tiny stickers.  
4. Sticker boxes from crops are **remapped** back to original image coordinates.  
5. Performance is evaluated with [pycocotools](https://pypi.org/project/pycocotools/) **COCOeval**.

---
## Implementations
The study has two implementations of WiSDet, one using [FsDet](https://github.com/ucbdrive/few-shot-object-detection) and another using Ultralytics [YOLOv8](https://docs.ultralytics.com/models/yolov8/).

Each model family has its own **`README`** in the repository:
[FsDet](FsDet/README.md)
[YOLOv8](YOLOv8/README.md)

These contain links to their trained weights, configuration files, and detailed usage.
---

