# Dataset Provenance, Audit, and Ingestion Guide

## Mandatory Dataset Audit

| Field | Description / Value |
|---|---|
| **Dataset Name** | Driver Safety AI Telemetry & Public Video Benchmark Data |
| **Source** | Public Driver Drowsiness/Yawn Video Repositories (e.g. YawDD, NTHU-DDD) & `collect_data.py` Telemetry |
| **License** | Academic / Research Use |
| **Subjects** | Multi-subject driver recordings (Split by Subject ID to prevent data leakage) |
| **Data Representation** | 12-Dimensional Temporal Feature Vectors extracted per frame via MediaPipe |
| **Window Sequence** | 30 consecutive frames × 12 features (`shape: (N, 30, 12)`) |
| **Target Classes** | `0: ALERT`, `1: DROWSY`, `2: YAWNING`, `3: DISTRACTED` |
| **Original Labels** | Video state annotations (Normal driving, eyes closed/drowsy, mouth open/yawn, looking away) |
| **Class Mapping Strategy** | Direct semantic mapping without forcing incompatible labels. Unmapped/ambiguous classes discarded. |
| **Limitations** | Single driver per frame; requires sufficient facial lighting for MediaPipe landmark tracking. |

---

## Dataset Pipeline Architecture

```
RAW VIDEO / TELEMETRY
         ↓
MediaPipe Face Mesh (640×480)
         ↓
12-Dimensional Feature Extractor
(EAR_LEFT, EAR_RIGHT, MEAN_EAR, MAR, YAW, PITCH, ROLL, PERCLOS, BLINK_RATE, EYE_CLOSURE_DUR, MOUTH_OPEN_DUR, HEAD_MOTION_MAG)
         ↓
Sliding Window Generator (30 frames, stride 1)
         ↓
StandardScaler (Fitted on Training Split Only)
         ↓
PyTorch Dataset (Subject-Level Split)
```

---

## Data Collection & Ingestion Instructions

### 1. Recording Live Driver Telemetry via Webcam
You can record high-quality driver telemetry sequences directly using `src/data/collect_data.py`:

> **[MANDATORY ALERT SESSION DATA PROTOCOL]**  
> During `ALERT` recording sessions, participants **MUST explicitly include realistic normal eye blinks** (blinking naturally at 12–20 blinks/min; each blink lasting 0.1s–0.35s).  
> **Do NOT record unblinking stare sessions.** Explicitly including normal blinks in `ALERT` sessions ensures the neural network learns to distinguish transient eye closure from sustained drowsiness.

```bash
# Record ALERT driving telemetry (MUST include natural 12-20 blinks/min)
python src/data/collect_data.py --label ALERT --subject_id subject_01

# Record DROWSY driver telemetry
python src/data/collect_data.py --label DROWSY --subject_id subject_01

# Record YAWNING driver telemetry
python src/data/collect_data.py --label YAWNING --subject_id subject_01

# Record DISTRACTED driver telemetry (sustained head deviation)
python src/data/collect_data.py --label DISTRACTED --subject_id subject_01
```

### 2. External Video Dataset Processing
If using external public video datasets (e.g., YawDD or NTHU-DDD):
1. Place raw mp4/avi videos into `data/raw/<subject_id>/`.
2. Run feature extraction script to produce structured CSV telemetry in `data/processed/`.
3. Verify that `subject_id` is present in the CSV to enforce subject-level train/val/test splitting.
