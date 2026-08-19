"""
Eye Aspect Ratio (EAR) calculation module.
Implements standard 6-landmark EAR for left eye, right eye, and mean EAR.
"""

import numpy as np
from typing import Tuple
from src.features.face_landmarks import FaceLandmarkExtractor

def calculate_ear(eye_landmarks_3d: np.ndarray) -> float:
    """
    Calculate Eye Aspect Ratio (EAR) given 6 facial landmarks (in 2D or 3D).
    Formula: EAR = (||p2-p6|| + ||p3-p5||) / (2.0 * ||p1-p4||)
    
    Landmark ordering:
    p1: outer corner
    p2: top-left
    p3: top-right
    p4: inner corner
    p5: bottom-right
    p6: bottom-left
    """
    if eye_landmarks_3d.shape[0] != 6:
        raise ValueError(f"Expected 6 landmarks for EAR, got {eye_landmarks_3d.shape[0]}")

    p1, p2, p3, p4, p5, p6 = eye_landmarks_3d[:6, :2] # Use 2D (x, y) coordinates for distance

    # Compute Euclidean distances
    vertical_dist_1 = np.linalg.norm(p2 - p6)
    vertical_dist_2 = np.linalg.norm(p3 - p5)
    horizontal_dist = np.linalg.norm(p1 - p4)

    if horizontal_dist < 1e-6:
        return 0.0

    ear = (vertical_dist_1 + vertical_dist_2) / (2.0 * horizontal_dist)
    return float(ear)

def extract_eye_features(landmarks_3d: np.ndarray) -> Tuple[float, float, float]:
    """
    Extract EAR_LEFT, EAR_RIGHT, MEAN_EAR from full 3D face landmarks array.
    Returns: (ear_left, ear_right, mean_ear)
    """
    left_eye_pts = landmarks_3d[FaceLandmarkExtractor.LEFT_EYE]
    right_eye_pts = landmarks_3d[FaceLandmarkExtractor.RIGHT_EYE]

    ear_left = calculate_ear(left_eye_pts)
    ear_right = calculate_ear(right_eye_pts)
    mean_ear = (ear_left + ear_right) / 2.0

    return ear_left, ear_right, mean_ear
