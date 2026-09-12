# 👁️ Real-Time Object Detection

### Give a webcam the ability to see.

A camera gives us pixels.

Computer vision gives those pixels meaning.

This project turns a live webcam stream into a real-time detection pipeline using **OpenCV's DNN module** and a pretrained **MobileNet-SSD** model.

As objects enter the camera's view, the system attempts to identify them and draws a bounding box with its confidence score.

```text
                📷 WEBCAM
                    │
                    ▼
             ┌─────────────┐
             │Frame Capture│
             └──────┬──────┘
                    │
                    ▼
             ┌─────────────┐
             │Preprocessing│
             │  300 × 300  │
             └──────┬──────┘
                    │
                    ▼
             ┌─────────────┐
             │MobileNet-SSD│
             │  Inference  │
             └──────┬──────┘
                    │
                    ▼
             Confidence Filter
                    │
                    ▼
             Bounding Boxes
                    │
                    ▼
              🖥️ LIVE VIEW
```

---

## ⚡ The Project in One Sentence

> **A lightweight webcam-based computer vision application that performs real-time object detection using a pretrained MobileNet-SSD model.**

---

# 🧠 What Is Happening Behind the Camera?

Every webcam frame passes through the same small inference loop.

```text
Frame
 │
 ├── Flip horizontally
 │
 ├── Read height & width
 │
 ├── Resize → 300 × 300
 │
 ├── Convert → DNN Blob
 │
 ├── MobileNet-SSD Forward Pass
 │
 ├── Read detections
 │
 ├── Confidence > 50% ?
 │       │
 │      YES
 │       ▼
 │   Class + Box
 │
 └── Draw on frame
```

The process repeats continuously until the user presses **`q`**.

---

# 🔬 The Detection Pipeline

## 1. Capture

OpenCV connects to the system's default webcam:

```python
cap = cv2.VideoCapture(0)
```

The application requests a **1920 × 1080** capture resolution and processes the incoming frames.

For a more natural webcam experience, each frame is horizontally flipped.

---

## 2. Prepare the Frame

MobileNet-SSD expects a fixed-size input.

The live frame is therefore resized to:

```text
300 × 300
```

and converted into a DNN blob:

```python
blob = cv2.dnn.blobFromImage(
    cv2.resize(frame, (300, 300)),
    0.007843,
    (300, 300),
    127.5
)
```

Conceptually:

```text
1920 × 1080 Frame
       │
       ▼
   Resize
       │
       ▼
  300 × 300
       │
       ▼
  Normalized Blob
       │
       ▼
 Neural Network
```

---

# 🧩 Meet MobileNet-SSD

The project uses a pretrained **MobileNet-SSD** model in Caffe format.

Two model files are loaded:

```text
MobileNetSSD_deploy.prototxt
MobileNetSSD_deploy.caffemodel
```

The architecture combines:

**MobileNet**

→ a lightweight convolutional neural network architecture

with

**SSD — Single Shot Detector**

→ an object detection approach designed to predict object classes and bounding boxes in a single forward pass.

This makes the model well suited to lightweight real-time computer vision experiments.

---

# 🎯 Confidence Matters

The neural network produces multiple candidate detections.

Not every prediction is useful.

The project applies a simple threshold:

```python
if confidence > 0.5:
```

In other words:

```text
Prediction
    │
    ▼
Confidence Score
    │
    ├── < 50% ──→ Ignore
    │
    └── ≥ 50% ──→ Display
```

Only detections with confidence above **50%** are rendered on the live feed.

---

# 📦 From Prediction to Bounding Box

For every accepted detection, the application extracts:

* Object class
* Confidence
* Bounding-box coordinates

The normalized coordinates are converted back into the dimensions of the original webcam frame.

```text
              ┌──────────────────────────┐
              │                          │
              │       ┌──────────┐       │
              │       │   DOG    │       │
              │       │  87%     │       │
              │       └──────────┘       │
              │                          │
              └──────────────────────────┘
```

The result is rendered directly on the video frame.

---

# 👀 What Can It Detect?

The application maintains a class-label list containing common objects such as:

```text
✦ Person
✦ Car
✦ Bicycle
✦ Bird
✦ Boat
✦ Bottle
✦ Bus
✦ Cat
✦ Chair
✦ Cow
✦ Dog
✦ Horse
✦ Motorbike
✦ Potted Plant
✦ Sheep
✦ Sofa
✦ Train
✦ TV Monitor
```

The repository's label list also contains additional labels such as **book, phone, and headphones**.

> Detection quality depends on whether the underlying pretrained model actually supports a given class. Adding a label to the Python list does not itself train the model to recognize a new object.

---

# 🏗️ Architecture

Unlike a traditional web application, this project doesn't need a database, API server, or frontend framework.

Its architecture is intentionally compact:

```text
┌──────────────────┐
│     Webcam       │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ OpenCV Capture   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Frame Processing │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  OpenCV DNN      │
│  MobileNet-SSD   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Detection Filter │
│   > 50%          │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Bounding Boxes + │
│ Labels + Scores  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   Live Window    │
└──────────────────┘
```

---

# 🛠️ Technology Stack

| Technology       | Role                                    |
| ---------------- | --------------------------------------- |
| 🐍 Python        | Application logic                       |
| 👁️ OpenCV       | Computer vision & webcam processing     |
| 🧠 OpenCV DNN    | Neural-network inference                |
| 📦 MobileNet-SSD | Pretrained object detector              |
| 🔢 NumPy         | Numerical operations / array processing |
| ☕ Caffe Model    | Model format                            |

---

# 📂 Repository Layout

```text
real-time-object-detection/
│
├── object_detection.py
│
├── MobileNetSSD_deploy.prototxt
│
├── MobileNetSSD_deploy.caffemodel
│
├── requirements.txt
│
└── README.md
```

The `.caffemodel` contains the pretrained neural-network weights, while the `.prototxt` defines the model architecture/configuration.

---

# 🚀 Getting Started

## 1. Clone the repository

```bash
git clone https://github.com/taniiishaa/real-time-object-detection.git
cd real-time-object-detection
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

The project requires:

```text
opencv-python
numpy
```

## 3. Start the detector

```bash
python object_detection.py
```

Make sure a webcam is available and accessible to OpenCV.

---

# 🎮 Controls

There is intentionally only one interaction:

```text
Press Q
   ↓
Stop detection
   ↓
Release webcam
   ↓
Close OpenCV window
```

The application also performs cleanup with:

```python
cap.release()
cv2.destroyAllWindows()
```

---

# 📐 Why MobileNet-SSD?

This project is less about achieving state-of-the-art detection accuracy and more about understanding **how a pretrained deep-learning model can be integrated into a real application**.

MobileNet-SSD is particularly useful for learning because the pipeline is relatively lightweight:

```text
Pretrained Model
      +
OpenCV DNN
      +
Webcam
      ↓
Real-Time Detection
```

That makes it a good bridge between:

**Deep Learning Theory → Model Inference → Computer Vision Application**

---

# 💡 What This Project Demonstrates

This small application touches several important computer-vision concepts:

### Computer Vision

Working with live frames, resizing, image manipulation and drawing.

### Deep Learning Inference

Loading and running a pretrained neural-network model.

### Object Detection

Working with class probabilities and bounding boxes rather than simply classifying an entire image.

### Real-Time Processing

Running inference repeatedly over a live video stream.

### Model Integration

Using a pretrained model inside a Python application without training a neural network from scratch.

---

# 🌍 Where This Concept Can Go

The current application is intentionally simple, but the same detection pipeline can become the foundation for much larger systems.

```text
                  OBJECT DETECTION
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
   Object Count     Video Analytics   Smart Camera
        │                │                │
        ▼                ▼                ▼
   Statistics       Events/Alerts     Monitoring
```

Possible extensions include:

* 🎥 Video-file detection
* 📹 IP-camera support
* 🔢 Real-time object counting
* 📊 Detection statistics
* 🚨 Rule-based alerts
* 🖥️ Streamlit interface
* 🌐 Flask/FastAPI inference API
* ⚡ GPU acceleration
* 🎯 Custom-trained object detection
* 🗃️ Detection logging
* 📍 Region-of-interest monitoring

---

# ⚠️ Current Scope

This repository is a **real-time inference project**, not a model-training pipeline.

It currently:

* Uses a pretrained MobileNet-SSD model.
* Reads frames from the default webcam.
* Uses a 50% confidence threshold.
* Draws detections directly on the video.
* Does not store detection history.
* Does not perform object tracking.
* Does not provide a web dashboard.
* Does not train the detector on a custom dataset.

Keeping the scope clear makes the project easier to understand and technically honest.

---

# 🧪 A Useful Learning Experiment

One of the easiest ways to explore this project is to change the confidence threshold.

Current:

```python
if confidence > 0.5:
```

Try experimenting with:

```text
0.30
0.50
0.70
0.90
```

You'll observe the trade-off between:

**More detections ↔ More false positives**

and

**Fewer detections ↔ More conservative predictions**

That simple experiment is a practical introduction to threshold selection in machine-learning systems.

---

# 🚀 Future Vision

The natural evolution of this project would be:

```text
WEBCAM
   │
   ▼
DETECTION
   │
   ▼
TRACKING
   │
   ▼
COUNTING
   │
   ▼
EVENT DETECTION
   │
   ▼
ANALYTICS
   │
   ▼
REAL-WORLD DECISION SUPPORT
```

That's where a small computer-vision experiment can start becoming a genuine **AI-powered monitoring system**.

---

## ✨ Final Takeaway

> **The camera captures the world.
> The model turns pixels into objects.
> The application turns those predictions into something usable.**

This project is a compact demonstration of that entire journey — from a webcam frame to real-time deep-learning inference using Python and OpenCV.
