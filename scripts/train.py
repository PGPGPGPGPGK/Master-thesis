"""Train a YOLOv8 pedestrian-detection model.

Recovered and cleaned up from the thesis project's training script, which
had hard-coded local paths (C:/Users/pg/Desktop/...) and a fixed 3-epoch,
CPU-only run. Now takes arguments so it works on any machine/GPU.
"""

import argparse

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, default="argoverse.yaml", help="dataset YAML path")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="initial weights (pretrained checkpoint)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0", help="'cpu' or GPU index, e.g. '0'")
    args = parser.parse_args()

    model = YOLO(args.weights)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, device=args.device)


if __name__ == "__main__":
    main()
