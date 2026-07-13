# Pedestrian Detection for Autonomous Driving Safety (YOLOv8)

Reconstructed working pipeline from Pavanganesh Karnam's Master's thesis project
*"Improving Autonomous Driving Safety through Deep Learning-based Pedestrian
Detection Enhancement"* (SRH Berlin University of Applied Sciences, 2024).

## What this actually is

An early informal draft of the thesis described a YOLOv3 + TensorFlow 1.14 +
Caltech pipeline. The **final submitted thesis** (2024-03-24, matriculation
3107225) confirms what was actually run: **YOLOv8 (Ultralytics) trained on
the Argoverse dataset**, with image prediction and multi-object tracking on
top. This repo implements that actual pipeline.

Pipeline:

1. **`scripts/convert_argoverse_to_yolo.py`** — converts Argoverse-HD JSON
   annotations into YOLO-format label files.
2. **`scripts/train.py`** — fine-tunes a YOLOv8 model on the converted dataset.
3. **`scripts/predict.py`** — runs a trained checkpoint over a folder of
   images/frames.
4. **`scripts/track.py`** — runs YOLOv8's built-in tracker frame-by-frame via
   OpenCV, draws bounding boxes and motion trails per tracked pedestrian, and
   saves annotated frames.
5. **`scripts/track_video.py`** — a simpler alternative that tracks directly
   over a video file using Ultralytics' streaming API.

## Model

YOLOv8n (nano) — 225 layers, ~3.01M parameters, 8.2 GFLOPs. See
[`docs/images/model_summary.png`](docs/images/model_summary.png) for the
full layer-by-layer breakdown from an actual training run.

## Evidence this pipeline actually ran

The `docs/images/` folder holds editor screenshots recovered from the
original project directory, proving each part of the pipeline was executed
(not just described in the thesis text):

| File | What it shows |
|---|---|
| `model_summary.png` | Real YOLOv8n architecture printout from an actual training run |
| `class_names_config.png` | The exact class list used (`person`, `bicycle`, `car`, `motorcycle`, `bus`, `big car`, `traffic_light`, `stop_sign`) |
| `augmentation_config.png` | Training-time augmentation hyperparameters (flip, hue/saturation/exposure jitter) |
| `track_video_script.png` | The original streaming-tracker script that `scripts/track_video.py` is cleaned up from |

## Results (from the submitted thesis)

Trained and evaluated across 30 iterations with the SGD optimizer
(lr0=0.001, momentum=0.9, weight_decay=0.0001), converted Argoverse
annotations, and 8 classes (`person`, `bicycle`, `car`, `motorcycle`, `bus`,
`big car`, `traffic_light`, `stop_sign`).

**Best customized run:** Accuracy 0.85, Recall 0.75, F1 0.80.

**Effect of input image size** (3 epochs each):

| Image size | Precision | Recall | mAP50 | mAP50-95 | Inference time |
|---|---|---|---|---|---|
| 320×320 | 0.399 | 0.187 | 0.197 | 0.115 | 10.1 ms |
| 640×640 | 0.576 | 0.315 | 0.338 | 0.208 | 39.9 ms |

640×640 wins on every accuracy metric at ~4x the inference cost — the
thesis concludes 608–640px is close to optimal for this dataset/architecture.

Confusion matrices, F1/P/R/PR curves across iterations, and per-image
qualitative comparisons are in Chapter 8 of the thesis (not reproduced here
since they weren't recovered as standalone files).

## Status

- Steps 1–4 above were run during the thesis and formally evaluated (see
  Results above, and Chapter 7–8 of the submitted thesis).
- "Deployment and Testing" in the thesis refers to evaluation against a
  custom dataset, not physical PiCar deployment. The PiCar (Raspberry Pi
  car) integration was never completed — only generic RC-car motor/servo
  control code exists (see `third_party/`), never wired up to the trained
  pedestrian detector. That remains open future work.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# 1. Convert Argoverse-HD annotations to YOLO format
python scripts/convert_argoverse_to_yolo.py --dataset-dir /path/to/Argoverse

# 2. Train
python scripts/train.py --data argoverse.yaml --epochs 50 --imgsz 640

# 3. Predict on a folder of images
python scripts/predict.py --weights runs/detect/train/weights/best.pt --source /path/to/images

# 4. Track pedestrians in a video
python scripts/track.py --weights runs/detect/train/weights/best.pt --source /path/to/video.mp4
```

## third_party/

Contains a reference Raspberry Pi lane-keeping-assist project (MIT licensed,
© 2022 Shivam4797), used only as a structural reference for PiCar motor/servo
control. It is not pedestrian-detection code and is not original work from
this thesis — kept separate and attributed rather than merged into the main
pipeline.

## Dataset attribution

Argoverse-HD (ring-front-center camera), by Argo AI.
https://www.cs.cmu.edu/~mengtial/proj/streaming/
