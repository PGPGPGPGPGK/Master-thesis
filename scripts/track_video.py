"""Run YOLOv8 tracking directly over a video file and dump every frame to disk.

Simpler alternative to track.py, using Ultralytics' built-in streaming
tracker instead of a manual OpenCV loop.
"""

import argparse
import os

import cv2
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=str, required=True, help="path to trained .pt checkpoint")
    parser.add_argument("--source", type=str, required=True, help="path to video file")
    parser.add_argument("--output-dir", type=str, default="output_frames")
    parser.add_argument("--conf", type=float, default=0.3, help="confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model = YOLO(args.weights)
    results = model.track(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        show=False,
        stream=True,
        save=True,
        save_frames=True,
    )

    for frame_idx, result in enumerate(results):
        annotated = result.plot()
        frame_path = os.path.join(args.output_dir, f"frame_{frame_idx}.jpg")
        cv2.imwrite(frame_path, annotated)

    print(f"Frames saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
