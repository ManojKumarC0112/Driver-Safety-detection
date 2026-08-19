"""
Rule-Based EAR/MAR Baseline Model for Academic Comparison (Section 28).
Provides a non-deep-learning baseline using heuristic threshold logic on temporal features.
"""

import numpy as np
from typing import Union, Tuple, List

class RuleBasedBaselineModel:
    """
    Heuristic rule-based classifier using static EAR, MAR, and Head Pose thresholds.
    Classifies into 4 classes: 0: ALERT, 1: DROWSY, 2: YAWNING, 3: DISTRACTED.
    """
    def __init__(
        self,
        ear_threshold: float = 0.21,
        mar_threshold: float = 0.55,
        yaw_threshold: float = 25.0,
        pitch_threshold: float = 20.0,
    ):
        self.ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold

    def predict_single_frame(self, feature_vec: np.ndarray) -> int:
        """
        Classify a single 12-dim feature vector.
        Feature indices:
          0: EAR_LEFT, 1: EAR_RIGHT, 2: MEAN_EAR, 3: MAR, 4: YAW, 5: PITCH, 6: ROLL...
        """
        mean_ear = feature_vec[2]
        mar = feature_vec[3]
        yaw = feature_vec[4]
        pitch = feature_vec[5]

        # Priority decision rules
        if mean_ear < self.ear_threshold:
            return 1 # DROWSY
        elif mar > self.mar_threshold:
            return 2 # YAWNING
        elif abs(yaw) > self.yaw_threshold or abs(pitch) > self.pitch_threshold:
            return 3 # DISTRACTED
        else:
            return 0 # ALERT

    def predict_sequence(self, sequence: np.ndarray) -> int:
        """
        Classify a (30, 12) temporal sequence by evaluating the last frame or majority vote.
        """
        if sequence.ndim == 2:
            # Use last frame in 30-frame sequence for baseline
            return self.predict_single_frame(sequence[-1, :])
        elif sequence.ndim == 3:
            # Batch of sequences (B, 30, 12)
            preds = [self.predict_single_frame(sequence[i, -1, :]) for i in range(sequence.shape[0])]
            return np.array(preds, dtype=np.int64)
        else:
            raise ValueError(f"Expected 2D or 3D numpy array, got shape {sequence.shape}")
