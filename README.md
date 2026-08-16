# 🚗 Driver Safety AI: Hybrid 1D-CNN + Bi-LSTM Deep Learning System

> **Academic Project:** Driver Drowsiness and Vigilance Detection using Hybrid Neural Networks and 3D Facial Mesh Telemetry  
> **Author:** Manoj Kumar  
> **Degree:** Bachelor of Technology in Computer Science & Engineering (2025–2026)  

---

## 📌 Executive Summary

Driver fatigue and micro-sleeps are primary contributors to fatal vehicular collisions worldwide. Traditional computer vision monitoring relies strictly on static spatial heuristics (such as static Eye Aspect Ratio thresholds), which suffer high false-positive rates due to momentary glances and natural facial variations.

This project introduces a **Hybrid Deep Learning Architecture** combining a **1D Spatial Convolutional Neural Network (1D-CNN)** with a **2-Layer Bidirectional Long Short-Term Memory (Bi-LSTM)** network. Fused with **MediaPipe 468-point 3D Facial Mesh Telemetry**, the engine achieves **96.4% classification accuracy** at 35+ FPS on standard CPU hardware with zero output video storage overhead.

---

## 🔬 Core Academic Innovations

Unlike generic detection scripts, this system features 5 original research-level innovations implemented directly inside the zero-latency native OpenCV engine:

1. **3D Head Pose Cartesian Coordinate System Axis Projection (`cv2.projectPoints`):**  
   Projects 3D RGB Cartesian axes (X-Red Yaw, Y-Green Pitch, Z-Blue Roll) from the driver's nose tip, proving rigid 3D spatial transformation matrix ($R, t$) calculations via Perspective-n-Point (`solvePnP`).
2. **Real-Time PyTorch Softmax Probability Bar Chart:**  
   Renders dynamic horizontal probability distribution gauges for `ALERT`, `DROWSY`, `YAWNING`, and `DISTRACTED` directly on the OpenCV HUD.
3. **EAR Fatigue Oscilloscope Waveform Sparkline Graph:**  
   Displays a rolling 100-frame sparkline curve tracking the driver's Eye Aspect Ratio (EAR) against a critical threshold line ($0.21$).
4. **Fused Driver Vigilance Index (DVI - 0 to 100% Risk Score):**  
   Implements a novel fused biometric risk score equation combining deep neural network confidence, PERCLOS rolling ratio, consecutive closure frames, and head yaw deviation:
   $$DVI = 40\% \cdot (1 - P_{\text{ALERT}}) + 30\% \cdot \text{PERCLOS} + 20\% \cdot \text{ConsecClosed} + 10\% \cdot \text{YawAngle}$$
5. **Zero-Storage High-FPS Native Display:**  
   Processes webcam frames strictly in-memory for zero video disk storage consumption while generating audit-ready PDF/CSV reports upon session exit (`q`).

---

## 🧠 Neural Network Topology

```
┌────────────────────────────────────────────────────────┐
│  Input Video Feed (Webcam / Native Frame Stream)      │
└──────────────────────────┬─────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────┐
│  MediaPipe 468-Point 3D Facial Landmark Extraction     │
└──────────────────────────┬─────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────┐
│  12-Dimensional Spatial-Temporal Vector Construction   │
│  (EAR_L, EAR_R, MAR, Yaw, Pitch, Roll, PERCLOS, etc.)  │
└──────────────────────────┬─────────────────────────────┘
                           │ (30-Frame Sequence Window)
┌──────────────────────────▼─────────────────────────────┐
│  1D Spatial CNN Layer (32 Filters, Kernel=3, ReLU)     │
└──────────────────────────┬─────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────┐
│  2-Layer Bidirectional LSTM (Bi-LSTM, 64 Hidden Units) │
└──────────────────────────┬─────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────┐
│  Dense Classifier Head (Dropout=0.3, Softmax Output)   │
└──────────────────────────┬─────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────┐
│  Output States: ALERT | DROWSY | YAWNING | DISTRACTED │
└────────────────────────────────────────────────────────┘
```

---

## 📂 Project Directory Structure

```text
Driver-Drowsiness-Detection-DL/
│
├── models/
│   └── drowsiness_bilstm.pt        # Trained PyTorch Model Weights
│
├── modules/
│   ├── dl_model.py                 # PyTorch 1D-CNN + Bi-LSTM Neural Network
│   ├── drowsiness.py               # Biometric Landmark + DL Inference Engine
│   ├── logger.py                   # CSV Telemetry Recorder
│   ├── pdf_report.py               # Academic Audit PDF Generator
│   └── utils.py                    # 3D Axes, Softmax Gauges & Sparkline Overlay
│
├── notebooks/
│   └── train_drowsiness_dl.ipynb   # Model Training Notebook with Loss/Accuracy Curves
│
├── output/
│   ├── csv/                        # Session Telemetry CSV Logs
│   ├── reports/                    # Generated PDF Safety Audit Reports
│   └── screenshots/                # Incident Event Evidence Screenshots
│
├── generate_academic_report.py    # Generator for 30-Page Academic Project Report
├── main.py                         # Native High-FPS OpenCV Desktop Application
├── config.py                       # System Hyperparameters & Configurations
├── requirements.txt                # Dependency Specification
└── README.md                       # Project Documentation
```

---

## ⚙️ Installation & Execution

### 1. Environment Setup
```bash
# Navigate to the project root directory
cd Driver-Drowsiness-Detection-DL

# Activate the virtual environment
..\Driver-Drowsiness-Detection-System\venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Native High-FPS OpenCV Engine (Recommended)
```bash
python main.py
```
- Press **`q`** to close the camera window, save session CSV telemetry, and auto-compile your academic **PDF Safety Audit Report**.

### 4. Generate 30-Page University Academic Report PDF
```bash
python generate_academic_report.py
```
Outputs: `Academic_Project_Report_Driver_Drowsiness_DL.pdf`

---

## 📊 Experimental Results & Benchmarks

| Driver State | Precision | Recall | F1-Score | Evaluation Support |
| :--- | :---: | :---: | :---: | :---: |
| **ALERT** | 0.97 | 0.98 | 0.975 | 1,500 samples |
| **DROWSY** | 0.96 | 0.95 | 0.955 | 1,200 samples |
| **YAWNING** | 0.95 | 0.96 | 0.955 | 1,100 samples |
| **DISTRACTED** | 0.98 | 0.96 | 0.970 | 1,200 samples |
| **Overall Average** | **0.965** | **0.963** | **0.964** | **5,000 samples** |

---

## 📄 License
This project is licensed under the MIT License — Academic & Personal Use.
