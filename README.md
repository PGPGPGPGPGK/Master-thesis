# Pedestrian Detection for Autonomous Driving Safety

A YOLOv8 pipeline for detecting pedestrians and other road objects (cyclists,
vehicles, traffic lights, stop signs) from a vehicle-mounted camera, trained
on the Argoverse dataset. Built for a Master's thesis on improving
autonomous driving safety through deep learning-based pedestrian detection.

## Pipeline

1. **`scripts/convert_argoverse_to_yolo.py`** — converts Argoverse JSON
   annotations into YOLO-format label files.
2. **`scripts/train.py`** — fine-tunes a YOLOv8 model on the converted dataset.
3. **`scripts/predict.py`** — runs a trained checkpoint over a folder of
   images/frames.
4. **`scripts/track.py`** — tracks pedestrians frame-by-frame in a video via
   OpenCV, drawing bounding boxes and motion trails per tracked object.
5. **`scripts/track_video.py`** — a simpler alternative that tracks directly
   over a video file using Ultralytics' streaming API.

## Model

YOLOv8n (nano) — 225 layers, ~3.01M parameters, 8.2 GFLOPs. Trained on 8
classes: `person`, `bicycle`, `car`, `motorcycle`, `bus`, `big car`,
`traffic_light`, `stop_sign`.

## Results

| Image size | Precision | Recall | mAP50 | mAP50-95 | Inference time |
|---|---|---|---|---|---|
| 320×320 | 0.399 | 0.187 | 0.197 | 0.115 | 10.1 ms |
| 640×640 | 0.576 | 0.315 | 0.338 | 0.208 | 39.9 ms |

Best run (SGD, lr0=0.001, momentum=0.9, weight_decay=0.0001): **0.85
accuracy, 0.75 recall, 0.80 F1**. 640×640 input gives the best accuracy
across the board at roughly 4x the inference cost of 320×320.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# 1. Convert Argoverse annotations to YOLO format
python scripts/convert_argoverse_to_yolo.py --dataset-dir /path/to/Argoverse

# 2. Train
python scripts/train.py --data argoverse.yaml --epochs 50 --imgsz 640

# 3. Predict on a folder of images
python scripts/predict.py --weights runs/detect/train/weights/best.pt --source /path/to/images

# 4. Track pedestrians in a video
python scripts/track.py --weights runs/detect/train/weights/best.pt --source /path/to/video.mp4
```

## third_party/

A reference Raspberry Pi lane-keeping-assist project (MIT licensed, © 2022
Shivam4797), used as a structural reference for motor/servo control on a
camera-equipped RC car.

## Dataset

[Argoverse](https://www.cs.cmu.edu/~mengtial/proj/streaming/) (ring-front-center
camera), by Argo AI.
