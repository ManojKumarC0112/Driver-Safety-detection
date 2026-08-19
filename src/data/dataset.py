"""
PyTorch Dataset & Subject-Level Split Management.
Implements subject/session level dataset splitting, data validation (NaN/Inf checks),
class weight computation, and StandardScaler fitting/normalization.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset, DataLoader

from src.utils.paths import MODEL_PATH, SCALER_PATH, load_config
from src.data.create_sequences import create_sequences_from_array, load_telemetry_csv, CLASS_MAP

class DriverSafetyDataset(Dataset):
    """
    PyTorch Dataset for 30x12 driver temporal feature sequences.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        X: shape (N, 30, 12) float32
        y: shape (N,) int64
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

def validate_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Data validation suite (Section 21).
    Checks for NaN, Inf, invalid feature ranges, corrupted samples.
    """
    report = {
        "initial_samples": len(X),
        "nan_count": 0,
        "inf_count": 0,
        "discarded_samples": 0,
        "reasons": []
    }

    if len(X) == 0:
        report["status"] = "EMPTY"
        return X, y, report

    # Check for NaN / Inf
    nan_mask = np.isnan(X).any(axis=(1, 2))
    inf_mask = np.isinf(X).any(axis=(1, 2))
    invalid_mask = nan_mask | inf_mask

    report["nan_count"] = int(np.sum(nan_mask))
    report["inf_count"] = int(np.sum(inf_mask))
    report["discarded_samples"] = int(np.sum(invalid_mask))

    if report["discarded_samples"] > 0:
        report["reasons"].append(f"Discarded {report['discarded_samples']} sequences due to NaN/Inf values.")
        X = X[~invalid_mask]
        y = y[~invalid_mask]

    report["final_samples"] = len(X)
    report["status"] = "VALID"
    return X, y, report

def fit_and_save_scaler(
    X_train: np.ndarray,
    scaler_path: str = str(SCALER_PATH)
) -> StandardScaler:
    """
    Fit StandardScaler ONLY on training set (Section 15).
    X_train shape: (N, 30, 12)
    Reshapes to (N * 30, 12), fits scaler, and saves to file.
    """
    N, L, C = X_train.shape
    X_flat = X_train.reshape(-1, C)

    scaler = StandardScaler()
    scaler.fit(X_flat)

    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"[Dataset] Fitted StandardScaler saved to {scaler_path}")
    return scaler

def transform_with_scaler(
    X: np.ndarray,
    scaler: StandardScaler
) -> np.ndarray:
    """
    Transform sequence data using fitted StandardScaler without fitting.
    """
    N, L, C = X.shape
    X_flat = X.reshape(-1, C)
    X_norm_flat = scaler.transform(X_flat)
    return X_norm_flat.reshape(N, L, C).astype(np.float32)

def compute_class_weights(y_train: np.ndarray, num_classes: int = 4) -> torch.Tensor:
    """
    Compute class weights for handling class imbalance in CrossEntropyLoss (Section 22).
    """
    classes, counts = np.unique(y_train, return_counts=True)
    total_samples = len(y_train)

    weights = np.ones(num_classes, dtype=np.float32)
    for c, count in zip(classes, counts):
        if count > 0:
            weights[c] = total_samples / (num_classes * count)

    return torch.tensor(weights, dtype=torch.float32)

def prepare_subject_split_dataset(
    csv_paths: List[str],
    train_subjects: List[str],
    val_subjects: List[str],
    test_subjects: List[str],
    window_size: int = 30,
    stride: int = 1,
    scaler_path: str = str(SCALER_PATH)
) -> Tuple[DriverSafetyDataset, DriverSafetyDataset, DriverSafetyDataset, StandardScaler, torch.Tensor]:
    """
    Load telemetry CSVs, group by subject ID, split programmatically by subject to prevent data leakage,
    validate data, fit scaler on train set only, and return PyTorch Datasets.
    """
    train_X, train_y = [], []
    val_X, val_y = [], []
    test_X, test_y = [], []

    for path in csv_paths:
        feature_matrix, labels, subject_id, session_id = load_telemetry_csv(path)
        X_seq, y_seq = create_sequences_from_array(feature_matrix, labels, window_size, stride)

        if len(X_seq) == 0:
            continue

        if subject_id in train_subjects:
            train_X.append(X_seq)
            train_y.append(y_seq)
        elif subject_id in val_subjects:
            val_X.append(X_seq)
            val_y.append(y_seq)
        elif subject_id in test_subjects:
            test_X.append(X_seq)
            test_y.append(y_seq)
        else:
            # Default to train if subject not explicitly assigned
            train_X.append(X_seq)
            train_y.append(y_seq)

    if not train_X:
        raise ValueError("No training data found for specified train subjects.")

    X_train_raw = np.vstack(train_X)
    y_train = np.concatenate(train_y)

    X_val_raw = np.vstack(val_X) if val_X else np.empty((0, window_size, 12), dtype=np.float32)
    y_val = np.concatenate(val_y) if val_y else np.empty((0,), dtype=np.int64)

    X_test_raw = np.vstack(test_X) if test_X else np.empty((0, window_size, 12), dtype=np.float32)
    y_test = np.concatenate(test_y) if test_y else np.empty((0,), dtype=np.int64)

    # Validate
    X_train_raw, y_train, train_rep = validate_data(X_train_raw, y_train)
    if len(X_val_raw) > 0:
        X_val_raw, y_val, _ = validate_data(X_val_raw, y_val)
    if len(X_test_raw) > 0:
        X_test_raw, y_test, _ = validate_data(X_test_raw, y_test)

    # Fit Scaler ONLY on train
    scaler = fit_and_save_scaler(X_train_raw, scaler_path)

    # Transform
    X_train = transform_with_scaler(X_train_raw, scaler)
    X_val = transform_with_scaler(X_val_raw, scaler) if len(X_val_raw) > 0 else X_val_raw
    X_test = transform_with_scaler(X_test_raw, scaler) if len(X_test_raw) > 0 else X_test_raw

    class_weights = compute_class_weights(y_train)

    train_ds = DriverSafetyDataset(X_train, y_train)
    val_ds = DriverSafetyDataset(X_val, y_val)
    test_ds = DriverSafetyDataset(X_test, y_test)

    return train_ds, val_ds, test_ds, scaler, class_weights
