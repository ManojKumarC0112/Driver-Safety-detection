"""
Temporal Feature Aggregator.
Combines frame-level metrics (EAR, MAR, Head Pose) with temporal metrics (PERCLOS, Blink Rate, Durations, Motion)
to produce exact 12-dimensional feature vectors per frame.
"""

import math
import numpy as np
from collections import deque
from typing import Tuple, Dict, Any, Optional

from src.features.eye_features import extract_eye_features
from src.features.mouth_features import calculate_mar
from src.features.head_pose import HeadPoseEstimator

def wrap_angle_delta(curr_angle: float, prev_angle: float) -> float:
    """Calculates shortest angular distance between two angles in degrees (-180 to +180)."""
    delta = curr_angle - prev_angle
    return float(((delta + 180.0) % 360.0) - 180.0)

class TemporalFeatureExtractor:
    """
    Stateful temporal feature extractor tracking rolling metrics over frame buffers.
    Output: 12-dimensional vector:
      [EAR_LEFT, EAR_RIGHT, MEAN_EAR, MAR, YAW, PITCH, ROLL,
       PERCLOS, BLINK_RATE, EYE_CLOSURE_DURATION, MOUTH_OPEN_DURATION, HEAD_MOTION_MAGNITUDE]
    """

    FEATURE_NAMES = [
        "EAR_LEFT",
        "EAR_RIGHT",
        "MEAN_EAR",
        "MAR",
        "YAW",
        "PITCH",
        "ROLL",
        "PERCLOS",
        "BLINK_RATE",
        "EYE_CLOSURE_DURATION",
        "MOUTH_OPEN_DURATION",
        "HEAD_MOTION_MAGNITUDE",
    ]

    def __init__(
        self,
        ear_threshold: float = 0.21,
        mar_threshold: float = 0.55,
        perclos_window_frames: int = 90,
        blink_min_closed_frames: int = 2,
        blink_cooldown_frames: int = 5,
        fps: float = 30.0,
    ):
        self.ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold
        self.perclos_window_frames = perclos_window_frames
        self.blink_min_closed_frames = blink_min_closed_frames
        self.blink_cooldown_frames = blink_cooldown_frames
        self.fps = fps

        self.head_pose_estimator = HeadPoseEstimator()

        # Rolling state buffers
        self.eye_state_buffer = deque(maxlen=perclos_window_frames) # True if closed, False if open
        self.timestamp_buffer = deque(maxlen=perclos_window_frames)

        # Session tracking
        self.start_timestamp: Optional[float] = None

        # Blink state tracking
        self.blink_count = 0
        self.blink_history_timestamps = deque() # Timestamps of detected blinks within last 60s
        self.consecutive_closed_frames = 0
        self.blink_cooldown_timer = 0

        # Duration state tracking
        self.consecutive_mouth_open_frames = 0

        # Head motion tracking
        self.prev_yaw: Optional[float] = None
        self.prev_pitch: Optional[float] = None
        self.prev_roll: Optional[float] = None
        self.prev_timestamp: Optional[float] = None

    def reset(self):
        """Reset temporal state buffers."""
        self.eye_state_buffer.clear()
        self.timestamp_buffer.clear()
        self.blink_history_timestamps.clear()
        self.start_timestamp = None
        self.blink_count = 0
        self.consecutive_closed_frames = 0
        self.blink_cooldown_timer = 0
        self.consecutive_mouth_open_frames = 0
        self.prev_yaw = None
        self.prev_pitch = None
        self.prev_roll = None
        self.prev_timestamp = None

    def process_frame_landmarks(
        self,
        landmarks_3d: Optional[np.ndarray],
        is_valid_face: bool,
        frame_width: int,
        frame_height: int,
        timestamp_sec: float,
        fps: Optional[float] = None,
    ) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        """
        Process single-frame facial landmarks and aggregate into 12-dim vector.
        Returns:
            feature_vector (np.ndarray): Shape (12,)
            is_valid (bool): True if face was valid, False otherwise
            extra_info (dict): Raw values, rvec/tvec/cam_matrix for 3D axis visualization
        """
        if fps is not None and fps > 0:
            self.fps = fps

        if self.start_timestamp is None:
            self.start_timestamp = timestamp_sec

        extra_info = {
            "rvec": None,
            "tvec": None,
            "cam_matrix": None,
            "nose_2d": (0, 0),
            "is_blink": False,
            "status": "VALID" if is_valid_face else "INVALID",
        }

        if not is_valid_face or landmarks_3d is None:
            self.eye_state_buffer.append(False)
            self.timestamp_buffer.append(timestamp_sec)
            self.consecutive_closed_frames = 0
            self.consecutive_mouth_open_frames = 0
            
            feature_vec = np.zeros(12, dtype=np.float32)
            return feature_vec, False, extra_info

        # 1. Eye Aspect Ratio (EAR)
        ear_left, ear_right, mean_ear = extract_eye_features(landmarks_3d)

        # 2. Mouth Aspect Ratio (MAR)
        mar = calculate_mar(landmarks_3d)

        # 3. Head Pose Estimation (Yaw, Pitch, Roll)
        yaw, pitch, roll, rvec, tvec, cam_matrix = self.head_pose_estimator.estimate_pose(
            landmarks_3d, frame_width, frame_height
        )

        extra_info["rvec"] = rvec
        extra_info["tvec"] = tvec
        extra_info["cam_matrix"] = cam_matrix
        extra_info["nose_2d"] = (
            int(landmarks_3d[1, 0] * frame_width),
            int(landmarks_3d[1, 1] * frame_height),
        )

        # 4. PERCLOS (Fraction of closed eye frames in window)
        is_closed = mean_ear < self.ear_threshold
        self.eye_state_buffer.append(is_closed)
        self.timestamp_buffer.append(timestamp_sec)
        perclos = float(np.mean(self.eye_state_buffer)) if len(self.eye_state_buffer) > 0 else 0.0

        # 5. Blink Rate & Eye Closure Duration
        is_blink_event = False
        if is_closed:
            self.consecutive_closed_frames += 1
        else:
            if (
                self.consecutive_closed_frames >= self.blink_min_closed_frames
                and self.blink_cooldown_timer == 0
            ):
                self.blink_count += 1
                self.blink_history_timestamps.append(timestamp_sec)
                is_blink_event = True
                self.blink_cooldown_timer = self.blink_cooldown_frames

            self.consecutive_closed_frames = 0

        if self.blink_cooldown_timer > 0:
            self.blink_cooldown_timer -= 1

        extra_info["is_blink"] = is_blink_event

        # Calculate Blinks per Minute over rolling 60-second window (with minimum 10s warmup denominator)
        cutoff_time = timestamp_sec - 60.0
        while self.blink_history_timestamps and self.blink_history_timestamps[0] < cutoff_time:
            self.blink_history_timestamps.popleft()

        session_elapsed = timestamp_sec - self.start_timestamp
        effective_window = min(60.0, max(session_elapsed, 10.0))
        blink_rate = float(len(self.blink_history_timestamps) * (60.0 / effective_window))
        blink_rate = min(100.0, max(0.0, blink_rate)) # Cap physiologically

        eye_closure_duration = float(self.consecutive_closed_frames / max(self.fps, 1.0))

        # 6. Mouth Open Duration
        is_mouth_open = mar > self.mar_threshold
        if is_mouth_open:
            self.consecutive_mouth_open_frames += 1
        else:
            self.consecutive_mouth_open_frames = 0
        mouth_open_duration = float(self.consecutive_mouth_open_frames / max(self.fps, 1.0))

        # 7. Head Motion Magnitude (Angular Velocity in degrees/sec with angle wrapping)
        if (
            self.prev_yaw is not None 
            and self.prev_pitch is not None 
            and self.prev_roll is not None
            and self.prev_timestamp is not None
        ):
            dt = max(timestamp_sec - self.prev_timestamp, 1e-4)
            d_yaw = wrap_angle_delta(yaw, self.prev_yaw)
            d_pitch = wrap_angle_delta(pitch, self.prev_pitch)
            d_roll = wrap_angle_delta(roll, self.prev_roll)
            
            angular_dist = math.sqrt(d_yaw**2 + d_pitch**2 + d_roll**2)
            head_motion_magnitude = float(angular_dist / dt) # deg/sec
        else:
            head_motion_magnitude = 0.0

        self.prev_yaw = yaw
        self.prev_pitch = pitch
        self.prev_roll = roll
        self.prev_timestamp = timestamp_sec

        # Construct 12-dimensional feature vector
        feature_vec = np.array([
            ear_left,
            ear_right,
            mean_ear,
            mar,
            yaw,
            pitch,
            roll,
            perclos,
            blink_rate,
            eye_closure_duration,
            mouth_open_duration,
            head_motion_magnitude,
        ], dtype=np.float32)

        return feature_vec, True, extra_info
