"""
Sequence Generator Module.
Transforms continuous 12-feature frame telemetry into overlapping 30-frame temporal windows for PyTorch model input.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
from src.features.temporal_features import TemporalFeatureExtractor

# Class mapping constant
CLASS_MAP = {
    "ALERT": 0,
    "DROWSY": 1,
    "YAWNING": 2,
    "DISTRACTED": 3
}

REVERSE_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}

def create_sequences_from_array(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    window_size: int = 30,
    stride: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding window sequences from 2D numpy feature matrix (T, 12) and 1D label vector (T,).
    Returns:
        X (np.ndarray): Shape (N, window_size, 12)
        y (np.ndarray): Shape (N,) containing window-level target label
    """
    num_frames, num_features = feature_matrix.shape
    if num_features != 12:
        raise ValueError(f"Expected 12 features, got {num_features}")

    if num_frames < window_size:
        return np.empty((0, window_size, 12), dtype=np.float32), np.empty((0,), dtype=np.int64)

    X_list = []
    y_list = []

    for start in range(0, num_frames - window_size + 1, stride):
        end = start + window_size
        seq_features = feature_matrix[start:end, :] # Shape (window_size, 12)
        
        # Target label for window: mode or last frame's label
        seq_labels = labels[start:end]
        # Majority voting for window label
        vals, counts = np.unique(seq_labels, return_counts=True)
        majority_label = vals[np.argmax(counts)]

        X_list.append(seq_features)
        y_list.append(majority_label)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    return X, y

def load_telemetry_csv(
    csv_path: str,
    feature_cols: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray, str, str]:
    """
    Load telemetry CSV file and extract feature matrix, integer labels, subject_id, and session_id.
    """
    if feature_cols is None:
        feature_cols = TemporalFeatureExtractor.FEATURE_NAMES

    df = pd.read_csv(csv_path)

    # Check required columns
    for col in feature_cols + ["label", "subject_id", "session_id"]:
        if col not in df.columns:
            raise KeyError(f"Missing required column '{col}' in telemetry CSV {csv_path}")

    # Map string labels to integers
    df["label_id"] = df["label"].map(CLASS_MAP)
    if df["label_id"].isnull().any():
        invalid = df[df["label_id"].isnull()]["label"].unique()
        raise ValueError(f"Unrecognized class labels {invalid} in {csv_path}. Allowed: {list(CLASS_MAP.keys())}")

    feature_matrix = df[feature_cols].values.astype(np.float32)
    labels = df["label_id"].values.astype(np.int64)

    subject_id = str(df["subject_id"].iloc[0])
    session_id = str(df["session_id"].iloc[0])

    return feature_matrix, labels, subject_id, session_id
