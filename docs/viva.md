# Driver Safety AI: Viva Voce Preparation Guide

Comprehensive Questions & Answers for University Viva Voce Examination.

---

### Q1: What is the core objective of Driver Safety AI?
**A**: To build a non-intrusive real-time driver monitoring system that extracts facial landmark telemetry via MediaPipe, constructs 30-frame temporal sequences, and classifies driver state into four classes (`ALERT`, `DROWSY`, `YAWNING`, `DISTRACTED`) using a hybrid PyTorch 1D-CNN + 2-layer Bi-LSTM neural network.

---

### Q2: Why use MediaPipe Face Mesh instead of raw 2D image CNNs (like ResNet or VGG)?
**A**:
1. **Computational Latency**: Raw 2D CNNs processing high-resolution video frames consume significant GPU/CPU memory (~30-100 ms/frame). MediaPipe Face Mesh extracts 468 3D landmark points in ~8 ms on standard CPUs.
2. **Feature Efficiency**: Processing 12 numerical features per frame reduces the spatial dimensionality drastically while preserving essential facial dynamics (eye closure, mouth opening, head pose).

---

### Q3: Explain the 12 features extracted per frame.
**A**:
1-3. `EAR_LEFT`, `EAR_RIGHT`, `MEAN_EAR`: Eye Aspect Ratio measuring vertical vs horizontal eye opening.
4. `MAR`: Mouth Aspect Ratio measuring mouth opening vertical/horizontal ratio.
5. `YAW`: Head rotation around vertical axis (left/right tilt in degrees).
6. `PITCH`: Head rotation around transverse axis (up/down tilt in degrees).
7. `ROLL`: Head rotation around longitudinal axis (side tilt in degrees).
8. `PERCLOS`: Percentage of eye closure over rolling 90-frame window.
9. `BLINK_RATE`: Estimated blinks per minute over rolling 60-second window.
10. `EYE_CLOSURE_DURATION`: Consecutive seconds eyes remain closed.
11. `MOUTH_OPEN_DURATION`: Consecutive seconds mouth remains open.
12. `HEAD_MOTION_MAGNITUDE`: $\sqrt{\Delta Yaw^2 + \Delta Pitch^2 + \Delta Roll^2}$ tracking frame-to-frame movement speed.

---

### Q4: Explain the Eye Aspect Ratio (EAR) formula.
**A**: Using 6 3D landmarks per eye:
$$EAR = \frac{||p_2 - p_6|| + ||p_3 - p_5||}{2 \times ||p_1 - p_4||}$$
Where $p_1, p_4$ are horizontal corners, and $(p_2, p_6)$ and $(p_3, p_5)$ are vertical landmark pairs.

---

### Q5: What is PERCLOS and how is it calculated?
**A**: PERCLOS (Percentage of Eye Closure) measures the fraction of frames in a rolling observation window (e.g. 90 frames / 3 seconds) where MEAN_EAR falls below the closure threshold (0.21). It is a key physiological indicator of driver drowsiness.

---

### Q6: How is head pose estimated?
**A**: Using OpenCV `cv2.solvePnP` with 6 key 3D facial landmark points (Nose tip, Chin, Left eye outer corner, Right eye outer corner, Left mouth corner, Right mouth corner) matched against a 3D canonical facial model. `solvePnP` outputs rotation vectors (`rvec`) converted via Rodrigues transformation into Euler angles: Yaw, Pitch, Roll in degrees.

---

### Q7: Explain the 1D-CNN + 2-Layer Bi-LSTM Neural Network Architecture.
**A**:
- **Input Tensor**: `(Batch, 30, 12)`
- **Conv1D**: Input channels = 12, Output filters = 32, Kernel size = 3. Operates on temporal length to extract local feature patterns.
- **Bi-LSTM Layer**: 2 layers of Bidirectional LSTM with 64 hidden units. Because it is bidirectional, it outputs $64 \times 2 = 128$ features per temporal step `(Batch, 30, 128)`.
- **Dropout (0.3)**: Prevents overfitting on sequence representations.
- **Dense Classifier**: Linear layer mapping 128 sequence features to 4 raw class logits.

---

### Q8: What is the exact tensor transformation flow through PyTorch layers?
**A**:
1. Input: `[B, 30, 12]`
2. Transpose for Conv1d: `[B, 12, 30]`
3. Conv1d(12, 32, kernel_size=3, padding=1) + ReLU: `[B, 32, 30]`
4. Transpose for Bi-LSTM: `[B, 30, 32]`
5. Bi-LSTM(32, 64, num_layers=2, batch_first=True, bidirectional=True): `[B, 30, 128]`
6. Final step extraction: `[B, 128]`
7. Dropout(0.3) → Linear(128, 4): `[B, 4]` raw logits.

---

### Q9: Why use raw logits for training instead of Softmax?
**A**: PyTorch `nn.CrossEntropyLoss` internally combines `LogSoftmax` and `NLLLoss` for numerical stability. Applying Softmax before `CrossEntropyLoss` causes vanishing gradients and numerical instability. Softmax is applied strictly during real-time inference via `predict_proba()`.

---

### Q10: How do you prevent data leakage during dataset splitting?
**A**: By splitting the dataset strictly at the **Subject ID / Session level** (Subject-level split). Sequences from Driver A are placed in Training, while Driver B is placed in Validation, and Driver C in Test. Randomly splitting overlapping sliding windows from the same video into train and test causes severe data leakage and artificially inflated performance.

---

### Q11: How is Feature Normalization performed safely?
**A**: `StandardScaler` is fitted **ONLY** on the training dataset. The fitted scaler is saved to `models/feature_scaler.pkl`. Validation, test, and live webcam sequences are transformed using the saved scaler without re-fitting, preventing data leakage.

---

### Q12: What is the Driver Vigilance Index (DVI)?
**A**: DVI is a project-defined 0–100 risk score combining model predictions and temporal metrics:
$$DVI = 100 \times \left(0.40 \times (1 - P_{ALERT}) + 0.30 \times PERCLOS + 0.20 \times NormEyeClosure + 0.10 \times NormYawDev\right)$$
Categorized into:
- 0 – 25: LOW
- 25 – 50: MODERATE
- 50 – 75: HIGH
- 75 – 100: CRITICAL

---

### Q13: How is real-time performance optimized for CPU?
**A**:
1. MediaPipe processes frames on a 640×480 resized copy.
2. PyTorch model uses `torch.no_grad()` and `model.eval()`.
3. Single loading of MediaPipe mesh and neural network models on application startup.
4. Fast 12-feature vector sequence processing achieving 25-30+ FPS on CPU.
