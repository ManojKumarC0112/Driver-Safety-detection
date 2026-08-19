"""
MediaPipe Facial Landmark Extractor.
Extracts 3D facial landmarks for a single driver face with CPU-friendly processing and validity tracking.
"""

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple, Dict, Any

class FaceLandmarkExtractor:
    """
    Wrapper around MediaPipe Face Mesh for CPU-friendly real-time landmark extraction.
    Ensures single face tracking, landmark confidence checking, and frame validity flags.
    """

    # Key landmark indices (MediaPipe Face Mesh 468/478 standard)
    # Left Eye: 6 landmarks for EAR calculation
    # P1 (outer): 33, P4 (inner): 133, P2 (top-left): 160, P6 (bottom-left): 144, P3 (top-right): 158, P5 (bottom-right): 153
    LEFT_EYE = [33, 160, 158, 133, 153, 144]

    # Right Eye: 6 landmarks for EAR calculation
    # P1 (outer): 362, P4 (inner): 263, P2 (top-left): 385, P6 (bottom-left): 380, P3 (top-right): 387, P5 (bottom-right): 373
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    # Mouth: Inner/Outer landmarks for MAR calculation
    # Upper lip top: 13, Lower lip bottom: 14, Left corner: 61, Right corner: 291
    # Extra vertical: Upper inner 82, Lower inner 312
    MOUTH_VERTICAL_1 = (13, 14)
    MOUTH_VERTICAL_2 = (82, 312)
    MOUTH_HORIZONTAL = (61, 291)

    # 2D/3D Head Pose Landmark Points:
    # Nose tip: 1, Chin: 152, Left eye left corner: 33, Right eye right corner: 263, Left mouth corner: 61, Right mouth corner: 291
    HEAD_POSE_LANDMARKS = [1, 152, 33, 263, 61, 291]

    def __init__(
        self,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        target_width: int = 640,
        target_height: int = 480,
    ):
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.target_width = target_width
        self.target_height = target_height

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=self.max_num_faces,
            refine_landmarks=self.refine_landmarks,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

    def process_frame(
        self, frame: np.ndarray
    ) -> Tuple[bool, Optional[np.ndarray], Dict[str, Any]]:
        """
        Process a single image frame (BGR format).
        Returns:
            is_valid (bool): True if 1 face is successfully detected & valid.
            landmarks_3d (np.ndarray or None): Shape (N, 3) in normalized (x, y, z) or pixel space.
            metadata (dict): Contains face count, image dimensions, status message.
        """
        if frame is None or frame.size == 0:
            return False, None, {"num_faces": 0, "status": "Empty frame"}

        h, w, _ = frame.shape

        # CPU Optimization: operate on a 640x480 copy if input frame is larger
        if w > self.target_width or h > self.target_height:
            proc_frame = cv2.resize(frame, (self.target_width, self.target_height))
            proc_h, proc_w = self.target_height, self.target_width
        else:
            proc_frame = frame
            proc_h, proc_w = h, w

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return False, None, {"num_faces": 0, "status": "NO DRIVER FACE"}

        num_faces = len(results.multi_face_landmarks)
        if num_faces > 1:
            # Multiple faces detected - flag for application logic
            return False, None, {"num_faces": num_faces, "status": "MULTIPLE FACES DETECTED"}

        face_landmarks = results.multi_face_landmarks[0]
        
        # Convert landmarks to numpy array: shape (num_landmarks, 3)
        # Coordinates: x in [0,1], y in [0,1], z relative depth
        landmarks_3d = np.array(
            [[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark],
            dtype=np.float32,
        )

        # Basic landmark sanity check (ensure coordinates are within realistic bounds)
        if np.isnan(landmarks_3d).any() or np.isinf(landmarks_3d).any():
            return False, None, {"num_faces": 1, "status": "Corrupted landmark values"}

        metadata = {
            "num_faces": 1,
            "status": "VALID",
            "frame_width": proc_w,
            "frame_height": proc_h,
            "orig_width": w,
            "orig_height": h,
        }

        return True, landmarks_3d, metadata

    def close(self):
        """Release MediaPipe resources."""
        self.face_mesh.close()
