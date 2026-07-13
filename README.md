# Pedestrian Detection for Autonomous Driving Safety (YOLOv8)

Reconstructed working pipeline from Pavanganesh Karnam's Master's thesis project
*"Improving Autonomous Driving Safety through Deep Learning-based Pedestrian
Detection Enhancement"* (SRH Berlin University of Applied Sciences, 2024).

## What this actually is

The thesis document describes a YOLOv3 + TensorFlow 1.14 + Caltech pipeline.
What was actually run (recovered from script screenshots and local files) is
different: **YOLOv8 (Ultralytics) trained on the Argoverse-HD dataset**, with
prediction and multi-object tracking on top. This repo reflects what was
actually executed, not the thesis prose.

Pipeline:

1. **`scripts/convert_argoverse_to_yolo.py`** — converts Argoverse-HD JSON
   annotations into YOLO-format label files.
2. **`scripts/train.py`** — fine-tunes a YOLOv8 model on the converted dataset.
3. **`scripts/predict.py`** — runs a trained checkpoint over a folder of
   images/frames.
4. **`scripts/track.py`** — runs YOLOv8's built-in tracker on video, draws
   bounding boxes and motion trails per tracked pedestrian, and saves
   annotated frames.

## Status

- Steps 1–4 above were run during the thesis (evidence: `runs/detect/train`,
  `train10`, `train20` checkpoints referenced in scripts; annotated output
  frames on disk).
- PiCar (Raspberry Pi car) deployment of the trained model was **not**
  completed — only generic RC-car motor/servo control code exists
  (see `third_party/`), and it was never wired up to the pedestrian detector.
  That integration is open future work, not something to claim as done.
- The thesis's Chapter 4 (Experimental Evaluation) was left empty in the
  written draft — no formal precision/recall/mAP numbers were recorded
  anywhere found on disk.

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
