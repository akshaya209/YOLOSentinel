"""
Robust Person Detection using YOLOv4 + OpenCV
============================================
Far more accurate than HOG. Uses a pre-trained YOLOv4 model.

SETUP (run once):
  pip install opencv-python numpy requests

  Then download model files:
    python person_detection.py --download
"""

import cv2
import numpy as np
import argparse
import os
import urllib.request
import sys

# ─── CONFIG ────────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.5   # Min confidence to count as a detection (0–1)
NMS_THRESHOLD        = 0.4   # Non-max suppression — reduces overlapping boxes
INPUT_SIZE           = 416   # YOLOv4 input size (416 or 608; 608 = more accurate, slower)
TARGET_CLASS         = "person"
# ───────────────────────────────────────────────────────────────────────────────

MODEL_FILES = {
    "cfg":     ("yolov4.cfg",     "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg"),
    "weights": ("yolov4.weights", "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights"),
    "names":   ("coco.names",     "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names"),
}


def download_models():
    """Download YOLOv4 model files if not present."""
    print("Downloading YOLOv4 model files...")
    for key, (filename, url) in MODEL_FILES.items():
        if os.path.exists(filename):
            print(f"  ✓ {filename} already exists, skipping.")
            continue
        print(f"  ↓ Downloading {filename} ...")
        try:
            def progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if total_size > 0:
                    pct = min(downloaded * 100 / total_size, 100)
                    sys.stdout.write(f"\r    {pct:.1f}%")
                    sys.stdout.flush()
            urllib.request.urlretrieve(url, filename, progress)
            print(f"\r    ✓ Done ({os.path.getsize(filename) // (1024*1024)} MB)")
        except Exception as e:
            print(f"\n    ✗ Failed: {e}")
            sys.exit(1)
    print("All files ready.\n")


def load_model():
    """Load YOLOv4 network and class names."""
    for key, (filename, _) in MODEL_FILES.items():
        if not os.path.exists(filename):
            print(f"Missing: {filename}\nRun:  python person_detection.py --download")
            sys.exit(1)

    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    net = cv2.dnn.readNetFromDarknet("yolov4.cfg", "yolov4.weights")

    # Use GPU if available (CUDA), else CPU
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA if cv2.cuda.getCudaEnabledDeviceCount() > 0
                             else cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA if cv2.cuda.getCudaEnabledDeviceCount() > 0
                            else cv2.dnn.DNN_TARGET_CPU)

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

    return net, output_layers, classes


def detect_persons(frame, net, output_layers, classes):
    """Run YOLO inference and return bounding boxes for persons only."""
    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        frame, 1 / 255.0, (INPUT_SIZE, INPUT_SIZE),
        swapRB=True, crop=False
    )
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes, confidences = [], []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])

            if classes[class_id] != TARGET_CLASS:
                continue
            if confidence < CONFIDENCE_THRESHOLD:
                continue

            # YOLO returns center x/y + width/height (normalised)
            cx, cy, bw, bh = (detection[:4] * np.array([w, h, w, h])).astype(int)
            x = cx - bw // 2
            y = cy - bh // 2

            boxes.append([x, y, bw, bh])
            confidences.append(confidence)

    # Non-max suppression to remove duplicate overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            results.append((boxes[i], confidences[i]))

    return results


def draw_detections(frame, detections):
    """Draw stylish bounding boxes and labels."""
    for (x, y, w, h), conf in detections:
        # Clamp to frame bounds
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)

        # Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 100), 2)

        # Corner accents
        corner_len = 15
        thickness  = 3
        color_corner = (0, 255, 130)
        for px, py, dx, dy in [
            (x1, y1, 1,  1), (x2, y1, -1,  1),
            (x1, y2, 1, -1), (x2, y2, -1, -1),
        ]:
            cv2.line(frame, (px, py), (px + dx * corner_len, py), color_corner, thickness)
            cv2.line(frame, (px, py), (px, py + dy * corner_len), color_corner, thickness)

        # Label background + text
        label = f"Person  {conf:.0%}"
        (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - lh - baseline - 6), (x1 + lw + 6, y1), (0, 180, 80), -1)
        cv2.putText(frame, label, (x1 + 3, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    return frame


def draw_hud(frame, count, fps):
    """Overlay HUD — count + FPS."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (220, 60), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, f"Persons : {count}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 100), 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS     : {fps:.1f}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Q  quit", (frame.shape[1] - 90, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)
    return frame


def run(source):
    net, output_layers, classes = load_model()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Cannot open source: {source}")
        sys.exit(1)

    print("Running… press Q to quit.")
    tick = cv2.getTickFrequency()

    while True:
        t0 = cv2.getTickCount()
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (960, 540))          # consistent display size
        detections = detect_persons(frame, net, output_layers, classes)
        draw_detections(frame, detections)

        fps = tick / (cv2.getTickCount() - t0)
        draw_hud(frame, len(detections), fps)

        cv2.imshow("Person Detection — YOLOv4", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ─── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robust person detection with YOLOv4")
    parser.add_argument("--download", action="store_true", help="Download model files and exit")
    parser.add_argument("--source",   default=0,           help="Webcam index (0) or video file path")
    args = parser.parse_args()

    if args.download:
        download_models()
        print("Done. Run:  python person_detection.py")
        sys.exit(0)

    # Convert source to int if it's a digit string (webcam index)
    source = int(args.source) if str(args.source).isdigit() else args.source
    run(source)
