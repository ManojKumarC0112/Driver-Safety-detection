"""
Dataset Split & Data Leakage Verification Test Suite.
Verifies participant-level isolation, zero-leakage assertions, and minimum participant validation.
"""

import os
import shutil
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.data.dataset_builder import validate_participant_splits, build_proposed_dataset

def test_subject_split_no_leakage():
    train_subjects = ["subject_01", "subject_02", "subject_03"]
    val_subjects = ["subject_04"]
    test_subjects = ["subject_05"]

    # Must pass without raising ValueError
    validate_participant_splits(train_subjects, val_subjects, test_subjects)

def test_validate_participant_splits_rejection():
    """Confirms that builder rejects any split configuration where a participant appears in two splits."""
    train_subjects = ["P001", "P002"]
    val_subjects = ["P001"] # P001 overlaps with train!
    test_subjects = ["P003"]

    with pytest.raises(ValueError, match="Participant-level isolation violation detected"):
        validate_participant_splits(train_subjects, val_subjects, test_subjects)

    # Test val and test overlap
    with pytest.raises(ValueError, match="Participant-level isolation violation detected"):
        validate_participant_splits(["P001"], ["P002"], ["P002"])

def test_build_proposed_dataset_insufficient_participants():
    """Confirms that build_proposed_dataset raises ValueError when insufficient participants exist (<3)."""
    temp_dir = tempfile.mkdtemp()
    proc_dir = tempfile.mkdtemp()
    
    try:
        # Create dummy CSV files for only 2 participants
        feature_cols = [
            "EAR_LEFT", "EAR_RIGHT", "MEAN_EAR", "MAR",
            "YAW", "PITCH", "ROLL",
            "PERCLOS", "BLINK_RATE", "EYE_CLOSURE_DURATION", "MOUTH_OPEN_DURATION", "HEAD_MOTION_MAGNITUDE"
        ]
        
        for pid in ["P001", "P002"]:
            data = {"timestamp": np.linspace(0, 5, 40), "participant_id": pid, "session_id": "S001", "frame_id": range(40), "label": "ALERT"}
            for col in feature_cols:
                data[col] = 0.5
            df = pd.DataFrame(data)
            df.to_csv(os.path.join(temp_dir, f"{pid}_S001_ALERT.csv"), index=False)

        # Should raise ValueError due to < 3 participants
        with pytest.raises(ValueError, match="Insufficient participants for participant-isolated"):
            build_proposed_dataset(raw_dir=temp_dir, proc_dir=proc_dir, min_participants=3)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        shutil.rmtree(proc_dir, ignore_errors=True)

def test_build_proposed_dataset_valid_3_participants():
    """Confirms that build_proposed_dataset generates valid isolated splits when >= 3 participants exist."""
    temp_dir = tempfile.mkdtemp()
    proc_dir = tempfile.mkdtemp()
    
    try:
        feature_cols = [
            "EAR_LEFT", "EAR_RIGHT", "MEAN_EAR", "MAR",
            "YAW", "PITCH", "ROLL",
            "PERCLOS", "BLINK_RATE", "EYE_CLOSURE_DURATION", "MOUTH_OPEN_DURATION", "HEAD_MOTION_MAGNITUDE"
        ]
        
        for pid in ["P001", "P002", "P003"]:
            data = {"timestamp": np.linspace(0, 5, 40), "participant_id": pid, "session_id": "S001", "frame_id": range(40), "label": "ALERT"}
            for col in feature_cols:
                data[col] = 0.5
            df = pd.DataFrame(data)
            df.to_csv(os.path.join(temp_dir, f"{pid}_S001_ALERT.csv"), index=False)

        manifest = build_proposed_dataset(raw_dir=temp_dir, proc_dir=proc_dir, min_participants=3)
        assert manifest is not None
        assert len(manifest["train_participants"]) == 1
        assert len(manifest["validation_participants"]) == 1
        assert len(manifest["test_participants"]) == 1
        assert manifest["zero_leakage_verified"] is True

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        shutil.rmtree(proc_dir, ignore_errors=True)
