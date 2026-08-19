"""
Unit tests for Feature Engineering Module (EAR, MAR, Head Pose, Temporal Features).
Verifies angle wrapping, head pose ranges, blink rate warmup, PERCLOS, closure durations, and motion.
"""

import numpy as np
import pytest
from src.features.eye_features import calculate_ear, extract_eye_features
from src.features.mouth_features import calculate_mar
from src.features.head_pose import HeadPoseEstimator
from src.features.temporal_features import TemporalFeatureExtractor, wrap_angle_delta

def test_wrap_angle_delta():
    assert wrap_angle_delta(10.0, 5.0) == pytest.approx(5.0)
    assert wrap_angle_delta(-179.0, 179.0) == pytest.approx(2.0)
    assert wrap_angle_delta(179.0, -179.0) == pytest.approx(-2.0)
    assert wrap_angle_delta(0.0, 360.0) == pytest.approx(0.0)

def test_calculate_ear():
    eye_pts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 0.0],
        [3.0, 2.0, 0.0],
        [4.0, 0.0, 0.0],
        [3.0, -2.0, 0.0],
        [1.0, -2.0, 0.0]
    ], dtype=np.float32)

    ear = calculate_ear(eye_pts)
    assert pytest.approx(ear, 0.01) == 1.0

def test_calculate_mar():
    landmarks = np.zeros((478, 3), dtype=np.float32)
    landmarks[13] = [0.0, 1.0, 0.0]
    landmarks[14] = [0.0, -1.0, 0.0]
    landmarks[82] = [0.0, 1.0, 0.0]
    landmarks[312] = [0.0, -1.0, 0.0]
    landmarks[61] = [-2.0, 0.0, 0.0]
    landmarks[291] = [2.0, 0.0, 0.0]

    mar = calculate_mar(landmarks)
    assert pytest.approx(mar, 0.01) == 0.5

def test_head_pose_estimator_normal_range():
    estimator = HeadPoseEstimator()
    landmarks = np.zeros((478, 3), dtype=np.float32)
    landmarks[1] = [0.5, 0.5, 0.0]     # Nose
    landmarks[152] = [0.5, 0.8, 0.0]   # Chin
    landmarks[33] = [0.3, 0.4, 0.0]    # Left eye
    landmarks[263] = [0.7, 0.4, 0.0]   # Right eye
    landmarks[61] = [0.35, 0.7, 0.0]   # Left mouth
    landmarks[291] = [0.65, 0.7, 0.0]  # Right mouth

    yaw, pitch, roll, rvec, tvec, cam_matrix = estimator.estimate_pose(landmarks, 640, 480)
    assert isinstance(yaw, float)
    assert isinstance(pitch, float)
    assert isinstance(roll, float)
    assert rvec is not None
    
    # Pitch, Yaw, Roll for near-neutral face must be within [-90, +90]
    assert -90.0 <= pitch <= 90.0
    assert -90.0 <= yaw <= 90.0
    assert -90.0 <= roll <= 90.0

def test_temporal_feature_extractor_blink_rate_warmup():
    extractor = TemporalFeatureExtractor(fps=30.0)
    landmarks = np.zeros((478, 3), dtype=np.float32)
    landmarks[1] = [0.5, 0.5, 0.0]

    # Simulate 30 frames (1 second) of open eyes
    for frame_idx in range(30):
        t = frame_idx / 30.0
        feat, is_valid, _ = extractor.process_frame_landmarks(landmarks, True, 640, 480, timestamp_sec=t)
    
    # Blink rate during initial 1 second without blinks must be 0.0 (not inflated)
    blink_rate = feat[8]
    assert blink_rate == pytest.approx(0.0)
    assert blink_rate <= 100.0

def test_temporal_feature_prolonged_eye_closure_and_perclos():
    extractor = TemporalFeatureExtractor(ear_threshold=0.25, fps=30.0)
    
    # Mock landmarks for closed eye (EAR < 0.25)
    closed_landmarks = np.zeros((478, 3), dtype=np.float32)
    closed_landmarks[1] = [0.5, 0.5, 0.0]
    # Set eye landmarks flat (zero vertical distance -> EAR = 0)
    
    for frame_idx in range(60): # 2 seconds of closed eyes
        t = frame_idx / 30.0
        feat, _, _ = extractor.process_frame_landmarks(closed_landmarks, True, 640, 480, timestamp_sec=t)

    perclos = feat[7]
    closure_dur = feat[9]

    assert perclos == pytest.approx(1.0, 0.05)
    assert closure_dur > 1.5 # ~2.0 seconds

def test_temporal_feature_head_motion_magnitude():
    extractor = TemporalFeatureExtractor(fps=30.0)
    landmarks = np.zeros((478, 3), dtype=np.float32)
    landmarks[1] = [0.5, 0.5, 0.0]

    feat1, _, _ = extractor.process_frame_landmarks(landmarks, True, 640, 480, timestamp_sec=0.0)
    feat2, _, _ = extractor.process_frame_landmarks(landmarks, True, 640, 480, timestamp_sec=0.033)

    motion = feat2[11]
    assert motion >= 0.0
    assert motion < 1000.0 # No multi-hundred degree wrap-around spikes
