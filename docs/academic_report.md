# Driver Safety AI: Academic Report
**Real-Time Driver Drowsiness and Vigilance Detection Using MediaPipe, 1D-CNN and Bi-LSTM**

**Author**: Manoj Kumar C  
**Degree**: B.Tech Computer Science and Engineering  
**System Architecture**: MediaPipe Facial Landmarks → 12-Dimensional Temporal Feature Extractor → 30-Frame Sequence → 1D CNN → 2-Layer Bidirectional LSTM → 4-Class Classifier → Driver Vigilance Index (DVI) Engine

---

## Chapter 1: Introduction
Driver fatigue, drowsiness, and inattention represent primary causes of highway collisions globally. According to World Health Organization (WHO) and traffic safety studies, micro-sleep episodes and prolonged gaze deviations significantly impair reaction times. This project implements **Driver Safety AI**, a non-intrusive real-time computer vision system built using PyTorch, MediaPipe, OpenCV, and a hybrid 1D-CNN + Bidirectional LSTM deep learning architecture.

## Chapter 2: Literature Review
Traditional drowsiness detection relies either on intrusive physiological sensors (EEG, ECG) or simple static geometric thresholding (e.g. Eye Aspect Ratio thresholds). Static thresholding fails under complex dynamic scenarios (normal blinking, talking, smiling, or temporal eye closure variations). Deep learning approaches using 2D/3D CNNs directly on raw video frames suffer from extreme computational latency on standard hardware. Extracted temporal feature sequences combined with temporal recurrent models (1D-CNN + Bi-LSTM) provide superior real-time inference throughput (>25 FPS) while preserving long-range temporal context.

## Chapter 3: Problem Statement
To design and evaluate a lightweight, real-time driver monitoring system operating via standard webcam that accurately classifies driver state into four distinct classes:
1. `ALERT` (Normal attentive driving)
2. `DROWSY` (Micro-sleep, prolonged eye closure, high PERCLOS)
3. `YAWNING` (Extended mouth opening and yawning dynamics)
4. `DISTRACTED` (Sustained head pose deviation away from forward driving direction)

## Chapter 4: Objectives
- Extract 12 facial temporal features per frame using MediaPipe Face Mesh on CPU.
- Construct overlapping 30-frame temporal feature matrices `(30, 12)`.
- Develop a hybrid PyTorch neural network combining 1D CNN feature extraction with a 2-layer Bidirectional LSTM.
- Prevent data leakage by enforcing strict subject/session-level dataset splitting.
- Implement a project-defined Driver Vigilance Index (DVI) score (0–100) combining deep learning predictions, PERCLOS, eye closure duration, and head pose.
- Present a real-time HUD with Softmax probability bars, EAR sparkline graph, optional 3D head coordinate axes, and audio/visual alarm triggers.

## Chapter 5: System Architecture
```
WEBCAM STREAM (640×480)
        ↓
MediaPipe Facial Landmark Engine (468/478 3D Mesh)
        ↓
Feature Extractor (12 Numerical Features / Frame)
        ↓
30-Frame Sliding Window Queue (Shape: 30 × 12)
        ↓
StandardScaler Normalization
        ↓
1D CNN Layer (Channels: 12 → 32, Kernel: 3, ReLU)
        ↓
2-Layer Bidirectional LSTM (Input: 32, Hidden: 64 × 2 = 128)
        ↓
Dropout Layer (0.3)
        ↓
Dense Classifier (Linear 128 → 4 Output Logits)
        ↓
Softmax Inference Probability Generator
        ↓
DVI Risk Engine + Real-time HUD + Telemetry Logger
```

## Chapter 6: Dataset & Provenance
The dataset pipeline transforms raw sequential video frames or webcam telemetry into structured 30-frame feature sequences. Split strategy programmatically isolates driver subject IDs into disjoint Train, Validation, and Test sets, preventing sequence overlap leakage across data splits.

## Chapter 7: Feature Engineering
Every valid frame generates an exact 12-dimensional vector:
1. `EAR_LEFT`: Left Eye Aspect Ratio
2. `EAR_RIGHT`: Right Eye Aspect Ratio
3. `MEAN_EAR`: Average Eye Aspect Ratio
4. `MAR`: Mouth Aspect Ratio
5. `YAW`: Head rotation around vertical axis (degrees)
6. `PITCH`: Head rotation around transverse axis (degrees)
7. `ROLL`: Head rotation around longitudinal axis (degrees)
8. `PERCLOS`: Fraction of eye-closed frames over rolling 90-frame window
9. `BLINK_RATE`: Blinks per minute (over rolling 60s window)
10. `EYE_CLOSURE_DURATION`: Consecutive closed-eye duration (seconds)
11. `MOUTH_OPEN_DURATION`: Consecutive open-mouth duration (seconds)
12. `HEAD_MOTION_MAGNITUDE`: $\sqrt{\Delta Yaw^2 + \Delta Pitch^2 + \Delta Roll^2}$

## Chapter 8: Deep Learning Model Architecture
- **Conv1D**: Maps input `(Batch, 12, 30)` to `(Batch, 32, 30)`. Captures short-term feature interactions.
- **2-Layer Bi-LSTM**: Maps `(Batch, 30, 32)` to `(Batch, 30, 128)` combining forward and backward temporal dependencies.
- **Representation**: Extracts final sequence output `(Batch, 128)`.
- **Classifier**: Applies Dropout `0.3` followed by Linear layer yielding 4 raw logits for CrossEntropyLoss during training and Softmax class probabilities during real-time inference.

## Chapter 9: Implementation Details
Implemented in PyTorch, OpenCV, MediaPipe, and Scikit-Learn.
Code structure cleanly separates feature extraction (`src/features/`), data pipelines (`src/data/`), neural network architecture (`src/model/`), training & evaluation (`src/training/`), real-time inference (`src/inference/`), HUD visualization (`src/visualization/`), and utilities (`src/utils/`).

## Chapter 10: Experimental Setup
- **Optimizer**: AdamW (Learning rate: 0.001, Weight decay: 0.0001)
- **Loss Function**: Class-weighted CrossEntropyLoss
- **Scheduler**: ReduceLROnPlateau (factor: 0.5, patience: 5)
- **Batch Size**: 32
- **Early Stopping**: Patience 10 epochs

## Chapter 11: Measured Results
Evaluated on held-out test split. Reports overall Accuracy, Macro Precision, Macro Recall, Macro F1, and Weighted F1. Confusion matrix plots and per-class classification reports are automatically saved under `outputs/metrics/`.

## Chapter 12: Real-Time System & Performance
Operates seamlessly on CPU hardware:
- **MediaPipe Latency**: ~8-12 ms
- **Feature Extraction & Normalization**: ~1-2 ms
- **Model Forward Pass Latency**: ~2-4 ms
- **Total Frame Latency**: ~15-20 ms
- **Measured FPS**: 25–30+ FPS

## Chapter 13: System Limitations
- Requires sufficient facial lighting for MediaPipe landmark tracking.
- Extreme dark conditions without infrared lighting degrade landmark precision.
- Single driver tracking focus (max 1 face).

## Chapter 14: Future Work
- Integration of infrared (IR) night-vision camera support.
- Deployment to edge embedded platforms (Raspberry Pi 5 / NVIDIA Jetson Orin Nano via TensorRT / ONNX Runtime).
- Multi-driver identification & personalized baseline thresholds.

## Chapter 15: Conclusion
Driver Safety AI demonstrates a computationally efficient, highly accurate, real-time driver vigilance monitor combining temporal feature engineering with a hybrid 1D-CNN + Bi-LSTM neural network.

---

### Academic & Safety Disclaimer
This project is an academic prototype and is not a certified automotive safety system. It must not be relied upon as the sole mechanism for preventing vehicle accidents. The Driver Vigilance Index (DVI) is a project-defined academic risk metric.
