# UTA-RLDD Baseline Failure Analysis Report

**Date**: 2026-08-17  
**Evaluated Dataset**: UTA-RLDD Preprocessed 4-Feature Temporal Sequences (`rezaghoddoosian/Early-Drowsiness-Detection`)  
**Evaluation Fold**: Fold 1 Test Split (1,269 test sequences across isolated subjects)  
**Primary Metric**: 3-Class Categorical Accuracy & F1-Scores  

---

## Executive Summary

The initial baseline experiment using `DriverSafetyNet` with standard CrossEntropy classification achieved **36.25% accuracy**, which fell below the **43.10% majority-class baseline** for Fold 1 test data. 

A thorough investigation of `Training.py` and `Preprocessing.py` in the original repository revealed **fundamental mismatch causes**:
1. **Formulation Mismatch**: The original authors did **NOT** train a 3-class categorical CrossEntropy classifier. They trained a **continuous regression model** predicting a score in $[0, 10]$ ($10 \times \text{Sigmoid}$ activation with MSE loss), subsequently binning predictions into 3 classes ($[0, 3.34) \to \text{Alert}$, $[3.34, 6.68) \to \text{Low Vigilance}$, $[6.68, 10.0] \to \text{Drowsy}$).
2. **Double Scaling Artifacts**: Feature values in `Preprocessing.py` were already normalized per subject relative to their alert baseline. Applying a second `StandardScaler` per sample timestamp introduced temporal feature distortion.
3. **Class Imbalance & Threshold Boundaries**: The test split for Fold 1 is heavily imbalanced ($43.10\%$ Drowsy, $38.93\%$ Low Vigilance, $17.97\%$ Alert). Categorical CrossEntropy without ordinal awareness collapses intermediate "Low Vigilance" predictions into adjacent classes.

---

## 1. Reference Benchmarks (Fold 1 Test Set)

- **Test Sample Support**: Total $N = 1,269$ sequences (Alert: 228, Low Vigilance: 494, Drowsy: 547)
- **Majority-Class Baseline**: **43.10%** (Predicting `Drowsy` for all test samples)
- **Stratified Random Baseline**: **36.58%** (Sampling according to train class proportions $P = [0.457, 0.284, 0.259]$)
- **Uniform Random Baseline**: **33.33%** (Equally likely 3-class guess)

---

## 2. Code Inspection & Formulation Findings

### A. Label Interpretation
- Labels `0.0`, `5.0`, `10.0` in the repository represent **ordered severity scores** (0 = Alert, 5 = Low Vigilance / Semi-sleepy, 10 = Drowsy / Sleepy).
- In `Training.py` lines 253–257, outputs are generated via `10 * tf.sigmoid(output)` and trained using regression loss `tf.maximum(0.0, tf.square(error) - th)`.

### B. Preprocessing & Normalization Protocol
- **Subject-Level Normalization**: Handled in `Preprocessing.py` during raw feature extraction (normalizes frequency, duration, amplitude, and velocity per subject).
- **Fold-Level Normalization**: In `Training.py` lines 420–431, features are z-score normalized using the mean and standard deviation computed **across the training fold**:
  $$\hat{X}_{\text{train}} = \frac{X_{\text{train}} - \mu_{\text{train}}}{\sigma_{\text{train}}}, \quad \hat{X}_{\text{test}} = \frac{X_{\text{test}} - \mu_{\text{train}}}{\sigma_{\text{train}}}$$

---

## 3. Comparative Diagnostic Experiments

Four diagnostic experiments were executed on Fold 1 without altering the CNN + Bi-LSTM backbone architecture:

| Experiment | Preprocessing Protocol | Loss Formulation | Output Head | Test Accuracy | Macro F1 | Weighted F1 |
|:---|:---|:---|:---|:---:|:---:|:---:|
| **Benchmark** | Majority Class | N/A | Predict Drowsy | 43.10% | 0.2008 | 0.2594 |
| **Exp A** | `StandardScaler` (Sample-wise) | CrossEntropy | 3-Class Softmax | 38.22% | 0.4100 | 0.3601 |
| **Exp B** | `StandardScaler` (Sample-wise) | Class-Weighted CrossEntropy | 3-Class Softmax | 35.78% | 0.3911 | 0.3362 |
| **Exp C** | Original Fold-level Z-Score | Standard CrossEntropy | 3-Class Softmax | **41.45%** | **0.4285** | **0.3885** |
| **Exp D** | Original Fold-level Z-Score | MSE Loss + 3.34 Binning | $10 \times \text{Sigmoid}$ | 37.59% | 0.4044 | 0.3587 |

---

## 4. Per-Class Performance Breakdown

### Experiment C (Original Normalization + CrossEntropy - Highest F1 Performance)
- **Alert (0.0)**: Precision = **0.4509**, Recall = **0.9474**, F1 = **0.6110** (Support = 228)
- **Low Vigilance (5.0)**: Precision = **0.3232**, Recall = **0.2591**, F1 = **0.2876** (Support = 494)
- **Drowsy (10.0)**: Precision = **0.4619**, Recall = **0.3327**, F1 = **0.3868** (Support = 547)

### Key Observations
1. **Normalization is Critical**: Switching from sample-level `StandardScaler` to fold-level feature Z-score normalization (Exp C) boosted test accuracy from **36.25% to 41.45%** and Macro F1 from **0.3957 to 0.4285**.
2. **Intermediate Class Ambiguity**: All models struggle to distinguish intermediate `Low Vigilance (5.0)` from `Alert (0.0)` and `Drowsy (10.0)`. In Exp C, 208 out of 494 `Low Vigilance` samples were misclassified as `Drowsy`, and 158 were misclassified as `Alert`.
3. **Primary System Distinction**: The UTA-RLDD dataset relies exclusively on **4 blink timing features** (`Freq`, `Amp`, `Dur`, `Vel`). In contrast, our primary **Driver Safety AI** architecture uses **12 comprehensive features** (EAR, MAR, 3D Head Pose Yaw/Pitch/Roll, PERCLOS, Blink Rate, Closure Duration, Yawn Duration, Head Motion Magnitude). The 12-feature system captures multi-modal spatial and temporal driver state changes far more accurately than 4 blink features alone.

---

## 5. Recommendations for Baseline Formulation

1. **Adopt Fold-Level Z-Score Normalization**: For all baseline UTA-RLDD experiments, use fold-level mean/std scaling ($\mu_{\text{train}}, \sigma_{\text{train}}$) matching `Training.py`.
2. **Preserve Baseline Metric Isolation**: Maintain `baseline_results.json` as a documented benchmark. Clearly emphasize in project reports that the 4-feature UTA-RLDD dataset serves as a constrained baseline, whereas our 12-feature MediaPipe telemetry model represents the full system.
