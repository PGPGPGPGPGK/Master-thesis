"""Run a trained YOLOv8 checkpoint over a folder of images or video frames.

Recovered and cleaned up from the thesis project's prediction script, which
loaded a checkpoint from a hard-coded local path.
"""

import argparse

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=str, required=True, help="path to trained .pt checkpoint")
    parser.add_argument("--source", type=str, required=True, help="folder of images/frames, video, or image path")
    parser.add_argument("--save-frames", action="store_true", default=True)
    args = parser.parse_args()

    model = YOLO(args.weights)
    model(args.source, save=True, save_frames=args.save_frames)


if __name__ == "__main__":
    main()
