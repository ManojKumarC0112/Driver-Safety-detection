"""
Mouth Aspect Ratio (MAR) calculation module.
Implements MAR formula: vertical mouth distance / horizontal mouth distance.
"""

import numpy as np
from src.features.face_landmarks import FaceLandmarkExtractor

def calculate_mar(landmarks_3d: np.ndarray) -> float:
    """
    Calculate Mouth Aspect Ratio (MAR) given full face 3D landmarks array.
    Formula: MAR = (vertical_dist_1 + vertical_dist_2) / (2.0 * horizontal_dist)
    """
    p13 = landmarks_3d[13, :2]   # Upper lip center
    p14 = landmarks_3d[14, :2]   # Lower lip center
    p82 = landmarks_3d[82, :2]   # Inner upper lip
    p312 = landmarks_3d[312, :2] # Inner lower lip

    p61 = landmarks_3d[61, :2]   # Left mouth corner
    p291 = landmarks_3d[291, :2] # Right mouth corner

    vert_dist_1 = np.linalg.norm(p13 - p14)
    vert_dist_2 = np.linalg.norm(p82 - p312)
    horiz_dist = np.linalg.norm(p61 - p291)

    if horiz_dist < 1e-6:
        return 0.0

    mar = (vert_dist_1 + vert_dist_2) / (2.0 * horiz_dist)
    return float(mar)
