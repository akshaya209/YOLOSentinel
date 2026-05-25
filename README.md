# SentinelVision

Real-time human detection powered by **YOLOv4 + OpenCV**.

Fast. Accurate. Lightweight.

Detect people from:
- Webcam feeds
- CCTV streams
- Video files

Built with Python and optimized for real-time performance.

---

# Features

- Real-time person detection
- YOLOv4 accuracy
- CUDA GPU acceleration support
- Live FPS monitoring
- Person counting
- Clean HUD overlay
- Stylish bounding boxes
- Webcam + video file support
- Non-Max Suppression for cleaner detections

---

# Demo

## Live Detection
- Detects only humans
- Ignores other COCO classes
- Confidence-based filtering

---

# Tech Stack

- Python
- OpenCV DNN
- YOLOv4
- NumPy

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/yourusername/SentinelVision.git
cd SentinelVision
```

---

## 2. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install opencv-python numpy requests
```

---

# Download YOLOv4 Weights

Run once:

```bash
python3 person_detector.py --download
```

This downloads:
- `yolov4.cfg`
- `yolov4.weights`
- `coco.names`

---

# Usage

## Webcam Detection

```bash
python3 person_detector.py
```

---

## Video File Detection

```bash
python3 person_detector.py --source video.mp4
```

---

# Controls

| Key | Action |
|-----|--------|
| Q | Quit application |

---

# Project Structure

```text
SentinelVision/
│
├── person_detector.py
├── yolov4.cfg
├── yolov4.weights
├── coco.names
├── README.md
└── requirements.txt
```

---

# Performance

| Input Size | Speed | Accuracy |
|------------|------|----------|
| 320 | Fastest | Medium |
| 416 | Balanced | High |
| 608 | Slow | Very High |

Default:
```python
INPUT_SIZE = 416
```

---

# CUDA Acceleration

If OpenCV is compiled with CUDA support, SentinelVision automatically uses GPU acceleration.

```python
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
```

---

# Example Use Cases

- Smart surveillance systems
- Retail footfall analytics
- Restricted-area monitoring
- Crowd estimation
- Security automation
- Smart campus monitoring

---

# Future Improvements

- Multi-object tracking
- Person re-identification
- Email/SMS alerts
- Heatmap analytics
- Face recognition
- Web dashboard
- RTSP/IP camera support

---

# Troubleshooting

## OpenCV Import Error

```bash
pip install opencv-python
```

---

## Webcam Not Opening

Try another webcam index:

```bash
python3 person_detector.py --source 1
```

---

## Slow FPS

Lower input size:

```python
INPUT_SIZE = 320
```

---

# License

MIT License

---

# Author

Built with Python, OpenCV, and YOLOv4.
