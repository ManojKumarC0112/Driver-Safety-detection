# Driver Safety AI - 12-Feature Telemetry Semantics & Audit Specification

This document provides a formal mathematical and physical specification for the 12 telemetry features extracted by the real-time inference and data collection engine.

---

## 1. Feature Semantics & Audit Summary Table

| # | Feature | Mathematical Formula / Source | Units | Expected Range | Observed Range (Before Fix) | Root Cause / Audit Finding | Status |
|---|---|---|---|---|---|---|---|
| 1 | **EAR_LEFT** | $\frac{\|p_2-p_6\| + \|p_3-p_5\|}{2 \|p_1-p_4\|}$ | Ratio | $[0.0, 0.5]$ | $[0.14, 0.42]$ | Valid ratio computation. | **VERIFIED** |
| 2 | **EAR_RIGHT** | $\frac{\|p_2-p_6\| + \|p_3-p_5\|}{2 \|p_1-p_4\|}$ | Ratio | $[0.0, 0.5]$ | $[0.15, 0.43]$ | Valid ratio computation. | **VERIFIED** |
| 3 | **MEAN_EAR** | $\frac{\text{EAR}_{\text{LEFT}} + \text{EAR}_{\text{RIGHT}}}{2}$ | Ratio | $[0.0, 0.5]$ | $[0.15, 0.42]$ | Valid averaged eye ratio. | **VERIFIED** |
| 4 | **MAR** | $\frac{\|m_{13}-m_{14}\| + \|m_{82}-m_{312}\|}{2 \|m_{61}-m_{291}\|}$ | Ratio | $[0.0, 1.5]$ | $[0.08, 0.85]$ | Valid mouth aspect ratio. | **VERIFIED** |
| 5 | **YAW** | OpenCV `solvePnP` + `RQDecomp3x3` | Degrees ($^\circ$) | $[-90, +90]$ | $[-178.5, +179.8]$ | Flipped 3D model orientation caused $\pm 180^\circ$ wrap. Fixed with canonical camera frame coordinates. | **FIXED** |
| 6 | **PITCH** | OpenCV `solvePnP` + `RQDecomp3x3` | Degrees ($^\circ$) | $[-90, +90]$ | $[-179.9, +179.9]$ (Mean: $-105^\circ$) | $Y$-axis orientation inverted pitch to upside-down space. Fixed with $+Y$ pointing down to chin. | **FIXED** |
| 7 | **ROLL** | OpenCV `solvePnP` + `RQDecomp3x3` | Degrees ($^\circ$) | $[-90, +90]$ | $[-179.9, +179.7]$ | Gimbal offset from inverted pose. Fixed with `RQDecomp3x3` Euler decomposition. | **FIXED** |
| 8 | **PERCLOS** | $\frac{N_{\text{closed}}}{N_{\text{window}}}$ (90 frames) | Fraction $[0,1]$ | $[0.0, 1.0]$ | $[0.00, 1.00]$ | Rolling 3-second ratio of closed eyes ($\text{EAR} < 0.21$). | **VERIFIED** |
| 9 | **BLINK_RATE** | $N_{\text{blinks}} \times \frac{60.0}{\max(T_{\text{session}}, 10.0)}$ | Blinks / min | $[0, 60]$ | $[0.0, 662.6]$ | Extrapolating 1-second early windows inflated rate to 600+ blinks/min. Fixed with minimum 10s warmup denominator. | **FIXED** |
| 10 | **EYE_CLOSURE_DURATION** | Continuous closed frames $\times \frac{1}{\text{FPS}}$ | Seconds ($s$) | $[0.0, 30.0]$ | $[0.00, 12.40]$ | Continuous duration in seconds eyes remain closed ($<\text{threshold}$). Resets when eyes open. | **VERIFIED** |
| 11 | **MOUTH_OPEN_DURATION** | Continuous open frames $\times \frac{1}{\text{FPS}}$ | Seconds ($s$) | $[0.0, 30.0]$ | $[0.00, 15.20]$ | Continuous duration in seconds mouth remains open ($>\text{threshold}$). Resets when mouth closes. | **VERIFIED** |
| 12 | **HEAD_MOTION_MAGNITUDE** | $\frac{\sqrt{\Delta\text{Yaw}^2 + \Delta\text{Pitch}^2 + \Delta\text{Roll}^2}}{\Delta t}$ | Degrees / sec ($^\circ/s$) | $[0.0, 300.0]$ | $[0.0, 399.5]$ (Mean: $9.2$) | Angle wrap-around ($\pm 180^\circ$) produced $360^\circ$ delta spikes. Fixed with `wrap_angle_delta()` angular velocity. | **FIXED** |

---

## 2. Validation & Verification Status

- Unit Test Suite: `tests/test_features.py` (7/7 tests passing)
- Zero-Leakage Dataset Split: `tests/test_dataset_split.py` (4/4 tests passing)
- Full Integration Test Suite: 14/14 tests passing cleanly.
