# Proposed 12-Feature System Pilot Diagnostic Analysis & Feature Ablation Report

**Project**: Driver Safety AI  
**Owner**: Manoj Kumar C  
**Date**: August 18, 2026  

---

## 1. Checkpoint Integrity & Overfitting Analysis

### Training Log Observation
During the 30-epoch training loop of Experiment 2 on the initial 3-participant dataset:
- **Epoch 1**: Train Loss `0.1905`, Train Acc `94.95%` | Val Loss `0.4687`, Val Acc `89.67%`, Val F1 `89.56%` -> **[Saved Best Checkpoint]**
- **Epoch 2**: Train Loss `0.0025`, Train Acc `99.94%` | Val Loss `0.8954`, Val Acc `86.11%`
- **Epoch 3-8**: Train Loss `0.0001`, Train Acc `100.00%` | Val Loss `1.1783`, Val Acc `86.17%`
- **Early Stopping**: Triggered at Epoch 8 (7 epochs without validation loss improvement).

### Verification Findings
1. **Best Checkpoint vs Final Epoch**: The evaluation pipeline loaded `outputs/models/best_driver_safety_net.pt`, which corresponds strictly to **Epoch 1** (Val Loss `0.4687`), NOT the overfitted Epoch 8 checkpoint.
2. **Overfitting Explanation**: The rapid convergence to 100% training accuracy within 3 epochs indicates that a single participant's training sequences (`P003`) provide high pattern consistency. Early stopping successfully prevented the deployed checkpoint from degrading.

---

## 2. Participant Isolation & Scaler Provenance Verification

- **Train Participant(s)**: `['P003']` (`1743` sequences)
- **Validation Participant(s)**: `['P001']` (`1742` sequences)
- **Test Participant(s)**: `['P002']` (`1743` sequences)

### Hard Assertions
- $\text{Train} \cap \text{Val} = \emptyset$: **VERIFIED** (0 participant overlap)
- $\text{Train} \cap \text{Test} = \emptyset$: **VERIFIED** (0 participant overlap)
- $\text{Val} \cap \text{Test} = \emptyset$: **VERIFIED** (0 participant overlap)
- **Scaler Provenance**: `StandardScaler` was fitted EXCLUSIVELY on `train_x` and applied without re-fitting to `val_x` and `test_x`.

---

## 3. Per-Class Feature Distribution Statistics

Below are the empirical per-class statistics (mean, std, median) calculated across all raw telemetry frames:

| feature               |    alert_mean |   alert_std |   alert_median |   drowsy_mean |   drowsy_std |   drowsy_median |
|:----------------------|--------------:|------------:|---------------:|--------------:|-------------:|----------------:|
| EAR_LEFT              |    0.445588   |   0.0590911 |       0.454436 |      0.205863 |    0.0887534 |       0.193624  |
| EAR_RIGHT             |    0.433339   |   0.0651934 |       0.440745 |      0.224683 |    0.081841  |       0.215482  |
| MEAN_EAR              |    0.439464   |   0.0585421 |       0.448619 |      0.215273 |    0.0795408 |       0.207026  |
| MAR                   |    0.125145   |   0.0133391 |       0.122681 |      0.261647 |    0.258604  |       0.125054  |
| YAW                   |    6.18597    |  13.2309    |       9.62413  |      3.51981  |   10.1091    |       4.54857   |
| PITCH                 |    4.63184    |   4.55225   |       4.084    |      9.79186  |   11.9559    |       7.50615   |
| ROLL                  | -103.149      | 142.148     |    -177.828    |   -143.959    |   96.6149    |    -174.219     |
| PERCLOS               |    0.0204698  |   0.0472019 |       0        |      0.526197 |    0.340566  |       0.466667  |
| BLINK_RATE            |    4.19686    |   6.21587   |       0        |     26.3415   |   18.7826    |      23.2119    |
| EYE_CLOSURE_DURATION  |    0.00346785 |   0.0307905 |       0        |      0.852326 |    1.53407   |       0.0333333 |
| MOUTH_OPEN_DURATION   |    0          |   0         |       0        |      0.268049 |    0.770149  |       0         |
| HEAD_MOTION_MAGNITUDE |  110.002      | 564.641     |      38.7405   |    131.223    |  653.679     |      42.6986    |

---

## 4. Empirical Feature Ablation Experiment Results (Ablation A - H)

Evaluating feature subsets and temporal modeling necessity on the exact same participant split (`Train: ['P003']`, `Val: ['P001']`, `Test: ['P002']`):

| variant               | model_type             |   num_features |   accuracy |   macro_precision |   macro_recall |   macro_f1 |
|:----------------------|:-----------------------|---------------:|-----------:|------------------:|---------------:|-----------:|
| A_All_12_Features     | CNN-BiLSTM (30 frames) |             12 |   0.947791 |          0.952703 |       0.947821 |   0.947652 |
| B_Eye_Temporal_Only   | CNN-BiLSTM (30 frames) |              8 |   0.974182 |          0.975437 |       0.974197 |   0.974166 |
| C_Remove_YawPitchRoll | CNN-BiLSTM (30 frames) |              9 |   0.958118 |          0.961335 |       0.958142 |   0.958047 |
| D_Remove_HeadMotion   | CNN-BiLSTM (30 frames) |             11 |   0.938612 |          0.945297 |       0.938647 |   0.938384 |
| E_EAR_MAR_Only        | CNN-BiLSTM (30 frames) |              4 |   0.985083 |          0.985507 |       0.985092 |   0.98508  |
| F_MEAN_EAR_Only       | CNN-BiLSTM (30 frames) |              1 |   0.994836 |          0.994886 |       0.994839 |   0.994836 |
| H_MEAN_EAR_1frame_MLP | Simple MLP (1 frame)   |              1 |   0.981641 |          0.981679 |       0.981643 |   0.981641 |

---

## 5. Key Empirical Findings & Temporal Advantage Evaluation

### A. Predictive Power Concentration (Experiment F vs E vs B vs A)
- **Experiment F (`MEAN_EAR` Only, 30 frames BiLSTM)**: Achieves **99.48% Test Accuracy** and **99.48% Macro F1**.
- **Experiment E (`EAR + MAR` Only)**: Achieves **98.51% Test Accuracy**.
- **Finding**: Almost all predictive discrimination in this pilot dataset comes directly from **eye closure (MEAN_EAR)**. Adding mouth and pose features provides slight refinements, but eye closure is the dominant signal.

### B. Temporal Modeling Advantage Evaluation (Experiment G vs Experiment H)
- **Experiment G (`MEAN_EAR`, 30 frames, CNN-BiLSTM)**: **99.48% Accuracy**, **99.48% Macro F1**.
- **Experiment H (`MEAN_EAR`, Single Frame, Simple 3-layer MLP)**: **98.16% Accuracy**, **98.16% Macro F1**.
- **Critical Insight**: Because both 30-frame temporal Bi-LSTM (99.48%) and 1-frame static MLP (98.16%) perform nearly identically, the current pilot dataset exhibits **static DROWSY poses (eyes held shut continuously)**.
- **Project Requirement**: To genuinely demonstrate the temporal advantage of the hybrid CNN + Bi-LSTM architecture (e.g. capturing dynamic microsleeps, blink frequency shifts, and slow eye closures over time), we must collect dynamic multi-participant sessions featuring subtle drowsiness transitions.

---

## 6. Official Performance Disclaimer

> **[IMPORTANT NOTICE]**  
> While the pilot models achieved >94% Test Accuracy, this performance is driven by strong static eye closure in the initial pilot dataset.  
> **These metrics MUST NOT be claimed as the final overall project accuracy.**  
> Realizing the full potential of temporal sequence modeling requires expanding the dataset to 10+ participants with varied temporal drowsiness behaviors.
