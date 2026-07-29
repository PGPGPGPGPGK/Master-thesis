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

640×640 input gives the best accuracy across the board at roughly 4x the
inference cost of 320×320.

## Limitations & Next Steps

- **Recall trails precision.** At 640×640 the model still misses a
  meaningful share of true pedestrians (recall 0.315 vs. precision 0.576 in
  the reported run). For a safety-relevant detector, false negatives matter
  more than false positives, so recall — not just mAP — should be the
  primary metric to optimize going forward, e.g. via a lower confidence
  threshold, class-balanced loss weighting, or more pedestrian-heavy
  training data.
- **Short training runs.** Reported metrics come from 3-epoch runs, mainly
  to compare configurations cheaply. Full convergence would need
  significantly more epochs before these numbers are representative of the
  model's ceiling.
- **Nano model size.** YOLOv8n trades capacity for speed. A larger backbone
  (s/m) would likely close some of the recall gap at the cost of inference
  latency — worth benchmarking if the target hardware allows it.
- **Single dataset domain.** Training data is Argoverse (US driving
  scenes). Performance on other geographies/road layouts/weather is
  untested and would need a held-out domain-shifted validation set.
- **Latency vs. accuracy tradeoff unresolved.** 640×640 nearly quadruples
  inference time over 320×320 for a real accuracy gain; the right operating
  point depends on the target frame rate of the deployment hardware, which
  hasn't been fixed yet.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# 1. Convert Argoverse annotations to YOLO format
python scripts/convert_argoverse_to_yolo.py --dataset-dir /path/to/Argoverse

# 2. Train
python scripts/train.py --data data/argoverse.yaml --epochs 50 --imgsz 640

# 3. Predict on a folder of images
python scripts/predict.py --weights runs/detect/train/weights/best.pt --source /path/to/images

# 4. Track pedestrians in a video
python scripts/track.py --weights runs/detect/train/weights/best.pt --source /path/to/video.mp4
```

## Using your own dataset

You don't need Argoverse — the converter script is Argoverse-specific, but
training works with any dataset in standard YOLO layout:

```
my_dataset/
├── images/
│   ├── train/   # your training images (.jpg/.png)
│   └── val/     # your validation images
└── labels/
    ├── train/   # one .txt per image: "class x_center y_center width height" (normalized 0-1)
    └── val/
```

Create a dataset YAML (copy `data/argoverse.yaml` as a starting point) and
**replace the paths and class names with your own**:

```yaml
path: /path/to/my_dataset   # <-- your dataset root folder
train: images/train         # <-- your training images folder (relative to path)
val: images/val             # <-- your validation images folder

names:                      # <-- your own classes, numbered from 0
  0: person
  1: car
```

Then train with it:

```bash
python scripts/train.py --data /path/to/my_dataset.yaml --epochs 50 --imgsz 640
```

If your annotations are in another format (COCO JSON, Pascal VOC, Label
Studio), export them as "YOLO" from your labeling tool — most support it.

## Trained weights

The trained checkpoints (`.pt` files) are **not included in this repo** —
only the code is. To get a working model, either train one yourself with the
steps above (the `--weights yolov8n.pt` default downloads the pretrained
YOLOv8-nano base automatically), or use any YOLOv8 checkpoint you already
have with `predict.py` / `track.py`.

If you'd like the exact weights from the thesis experiments (`best.pt`,
trained on Argoverse), reach out to me on
[LinkedIn](https://www.linkedin.com/in/pavanganeshk) and I'll be happy to
share them.

## Dataset

[Argoverse](https://www.cs.cmu.edu/~mengtial/proj/streaming/) (ring-front-center
camera), by Argo AI.
