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
1. [FsDet](FsDet/README.md)
2. [YOLOv8](YOLOv8/README.md)
These contain links to their trained weights, configuration files, detailed setup, and usage.
---
## FsDet Overview


---
## YOLOv8 Overview


---
## Summary of Results
### Initial Models AP@50 Performance
| Model            |    31 shot |    10 shot |     5 shot |     2 shot |
| ---------------- | ---------: | ---------: | ---------: | ---------: |
| YOLOv8n          |     0.2524 |     0.1670 |     0.1551 |     0.1539 |
| YOLOv8l          | **0.4330** | **0.3839** | **0.3229** | **0.2874** |
| FsDet            |     0.2892 |     0.1407 |     0.1839 |     0.1773 |

### Models with WiSDet AP@50 Performance
| Model            |    31 shot |    10 shot |     5 shot |     2 shot |
| ---------------- | ---------: | ---------: | ---------: | ---------: |
| YOLOv8n + WiSDet |     0.4014 |     0.3350 |     0.3108 |     0.3147 |
| YOLOv8l + WiSDet |     0.4937 | **0.4200** |     0.3397 |     0.3049 |
| FsDet + WiSDet   | **0.5050** |     0.3180 | **0.3400** | **0.3320** |

### Windshield Detector AP@50 Performance
| Model   | AP@50 (windshield) |
| ------- | -----------------: |
| YOLOv8n |             0.8061 |
| YOLOv8l |             0.8770 |
| FsDet   |             0.9010 |
