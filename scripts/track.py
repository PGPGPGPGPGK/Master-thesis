"""Track pedestrians in a video with YOLOv8, drawing bounding boxes and
motion trails, and saving annotated frames.
"""

import argparse
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=str, required=True, help="path to trained .pt checkpoint")
    parser.add_argument("--source", type=str, required=True, help="path to video file (or 0 for webcam)")
    parser.add_argument("--trail-length", type=int, default=90, help="frames of trail history to retain per track")
    args = parser.parse_args()

    model = YOLO(args.weights)
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    track_history: dict[int, list[tuple[float, float]]] = defaultdict(list)
    frame_number = 0

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            results = model.track(frame, persist=True)

            # Get the boxes and track IDs
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Plot the tracks
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point
                if len(track) > args.trail_length:
                    track.pop(0)

                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

            # Display the annotated frame
            cv2.imshow("YOLOv8 Tracking", annotated_frame)

            # Save the annotated frame as an image
            cv2.imwrite(f"annotated_frame_{frame_number}.jpg", annotated_frame)
            frame_number += 1

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
