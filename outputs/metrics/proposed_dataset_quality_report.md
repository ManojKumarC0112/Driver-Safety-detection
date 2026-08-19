# Proposed 12-Feature Telemetry Dataset Quality Report

## 1. Dataset Overview
- **Dataset Name**: Driver Safety AI - Proposed 12-Feature Telemetry
- **Input Features**: 12 MediaPipe Telemetry Features
- **Target Classes**: `ALERT` (0) vs `DROWSY` (1)
- **Raw Session Files**: 6 telemetry CSVs
- **Total Participants**: 3 (`['P001', 'P002', 'P003']`)
- **Total Telemetry Frames**: 5402 frames
- **NaN / Inf Integrity**: 0 invalid entries (Pass)

## 2. Class Balance Breakdown
- **ALERT Frames**: 2701 frames
- **DROWSY Frames**: 2701 frames

## 3. 12-Feature Statistical Summary
| Feature | Min | Max | Mean | Std |
|:---|:---:|:---:|:---:|:---:|
| `EAR_LEFT` | 0.0157 | 0.6220 | 0.3257 | 0.1416 |
| `EAR_RIGHT` | 0.0601 | 1.0103 | 0.3290 | 0.1279 |
| `MEAN_EAR` | 0.0509 | 0.7131 | 0.3274 | 0.1321 |
| `MAR` | 0.1132 | 1.0828 | 0.1934 | 0.1954 |
| `YAW` | -71.9800 | 66.8193 | 4.8529 | 11.8470 |
| `PITCH` | -45.6205 | 39.5559 | 7.2119 | 9.4053 |
| `ROLL` | -179.9984 | 179.9977 | -123.5540 | 123.2118 |
| `PERCLOS` | 0.0000 | 1.0000 | 0.2733 | 0.3507 |
| `BLINK_RATE` | 0.0000 | 68.1796 | 15.2692 | 17.8392 |
| `EYE_CLOSURE_DURATION` | 0.0000 | 7.6000 | 0.4279 | 1.1648 |
| `MOUTH_OPEN_DURATION` | 0.0000 | 4.6333 | 0.1340 | 0.5607 |
| `HEAD_MOTION_MAGNITUDE` | 0.0000 | 9625.4805 | 120.6130 | 610.7638 |

## 4. Participant Isolation & Split Manifest
```json
{
  "train_participants": [
    "P003"
  ],
  "validation_participants": [
    "P001"
  ],
  "test_participants": [
    "P002"
  ],
  "train_sequence_count": 1743,
  "validation_sequence_count": 1742,
  "test_sequence_count": 1743,
  "feature_dimensions": [
    30,
    12
  ],
  "zero_leakage_verified": true
}
```