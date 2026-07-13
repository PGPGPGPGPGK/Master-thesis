"""Convert Argoverse-HD JSON annotations to YOLO-format label files.

Recovered and cleaned up from the thesis project's dataset-conversion script
(originally adapted from Ultralytics' argoverse2yolo helper, with
hard-coded local paths). Paths are now CLI arguments instead of being
hard-coded to a specific machine.
"""

import argparse
import json
from pathlib import Path

from tqdm import tqdm

# Native Argoverse-HD image resolution (ring-front-center camera).
IMG_WIDTH = 1920.0
IMG_HEIGHT = 1200.0


def argoverse_annotations_to_yolo(annotations_file: Path, images_root: Path) -> None:
    """Convert one Argoverse-HD annotations JSON file to per-image YOLO .txt labels."""
    labels: dict[str, list[str]] = {}
    data = json.load(open(annotations_file, "rb"))

    for annot in tqdm(data["annotations"], desc=f"Converting {annotations_file.name} to YOLO format..."):
        img_id = annot["image_id"]
        img_name = data["images"][img_id]["name"]
        img_label_name = f"{img_name[:-3]}txt"

        cls = annot["category_id"]
        x_center, y_center, width, height = annot["bbox"]
        x_center = (x_center + width / 2) / IMG_WIDTH
        y_center = (y_center + height / 2) / IMG_HEIGHT
        width /= IMG_WIDTH
        height /= IMG_HEIGHT

        seq_dir = data["seq_dirs"][data["images"][annot["image_id"]]["sid"]]
        img_dir = images_root / "labels" / seq_dir
        img_dir.mkdir(parents=True, exist_ok=True)

        key = str(img_dir / img_label_name)
        labels.setdefault(key, []).append(f"{cls} {x_center} {y_center} {width} {height}\n")

    for key, lines in labels.items():
        with open(key, "w") as f:
            f.writelines(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Argoverse dataset root (contains Argoverse-1.1/ and Argoverse-HD/annotations/)",
    )
    args = parser.parse_args()

    images_root = args.dataset_dir / "Argoverse-1.1"
    annotations_dir = args.dataset_dir / "Argoverse-HD" / "annotations"

    tracking_dir = images_root / "tracking"
    if tracking_dir.exists():
        tracking_dir.rename(images_root / "images")

    for split_file in ("train.json", "val.json"):
        argoverse_annotations_to_yolo(annotations_dir / split_file, images_root)


if __name__ == "__main__":
    main()
