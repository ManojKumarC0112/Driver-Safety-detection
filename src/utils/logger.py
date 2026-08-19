"""
Telemetry Logger & Session Summary Generator for Driver Safety AI.
Logs frame-level telemetry to CSV and compiles comprehensive session performance statistics upon session exit.
"""

import os
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from src.utils.paths import SESSIONS_DIR

class TelemetryLogger:
    """
    Stateful telemetry logger capturing per-frame features, probabilities, DVI, and performance metrics.
    Generates structured session telemetry CSV and summary JSON at exit.
    """
    def __init__(self, session_id: Optional[str] = None):
        if session_id is None:
            session_id = f"session_{int(time.time())}"
        self.session_id = session_id
        self.start_time = time.time()

        os.makedirs(SESSIONS_DIR, exist_ok=True)
        self.csv_path = SESSIONS_DIR / f"{self.session_id}_telemetry.csv"
        self.summary_json_path = SESSIONS_DIR / f"{self.session_id}_summary.json"

        self.columns = [
            "timestamp", "frame_id", "fps",
            "EAR_LEFT", "EAR_RIGHT", "MEAN_EAR", "MAR", "YAW", "PITCH", "ROLL",
            "PERCLOS", "BLINK_RATE", "EYE_CLOSURE_DURATION", "MOUTH_OPEN_DURATION", "HEAD_MOTION_MAGNITUDE",
            "P_ALERT", "P_DROWSY", "P_YAWNING", "P_DISTRACTED",
            "model_probability", "EAR", "closure_duration", "state", "alarm_triggered",
            "DVI", "predicted_state"
        ]

        self.records: List[Dict[str, Any]] = []
        self.fps_list: List[float] = []
        self.dvi_list: List[float] = []
        self.state_counts = {"ALERT": 0, "SUSPECT": 0, "DROWSY": 0, "RECOVERY": 0, "YAWNING": 0, "DISTRACTED": 0}
        self.alarm_events_count = 0
        self.blink_count = 0
        self.yawn_count = 0

    def log_frame(
        self,
        frame_id: int,
        fps: float,
        feature_vec: np.ndarray,
        probabilities: np.ndarray,
        dvi: float,
        predicted_state: str,
        is_blink: bool = False,
        is_yawn: bool = False,
        alarm_triggered: bool = False,
        state: Optional[str] = None,
        intervention_level: int = 0,
        voice_event: Optional[str] = None,
        response_status: str = "NONE"
    ):
        """Log telemetry for a single frame."""
        now_sec = time.time() - self.start_time
        final_state = state if state is not None else predicted_state

        row = {
            "timestamp": now_sec,
            "frame_id": frame_id,
            "fps": fps,
            "EAR_LEFT": float(feature_vec[0]),
            "EAR_RIGHT": float(feature_vec[1]),
            "MEAN_EAR": float(feature_vec[2]),
            "MAR": float(feature_vec[3]),
            "YAW": float(feature_vec[4]),
            "PITCH": float(feature_vec[5]),
            "ROLL": float(feature_vec[6]),
            "PERCLOS": float(feature_vec[7]),
            "BLINK_RATE": float(feature_vec[8]),
            "EYE_CLOSURE_DURATION": float(feature_vec[9]),
            "MOUTH_OPEN_DURATION": float(feature_vec[10]),
            "HEAD_MOTION_MAGNITUDE": float(feature_vec[11]),
            "P_ALERT": float(probabilities[0]),
            "P_DROWSY": float(probabilities[1]),
            "P_YAWNING": float(probabilities[2]),
            "P_DISTRACTED": float(probabilities[3]),
            "model_probability": float(probabilities[1]),
            "EAR": float(feature_vec[2]),
            "closure_duration": float(feature_vec[9]),
            "state": final_state,
            "alarm_triggered": alarm_triggered,
            "DVI": float(dvi),
            "predicted_state": predicted_state,
            "intervention_level": intervention_level,
            "voice_event": voice_event or "",
            "response_status": response_status
        }

        self.records.append(row)
        self.fps_list.append(fps)
        self.dvi_list.append(dvi)

        if predicted_state in self.state_counts:
            self.state_counts[predicted_state] += 1

        if is_blink:
            self.blink_count += 1
        if is_yawn:
            self.yawn_count += 1
        if alarm_triggered:
            self.alarm_events_count += 1

    def close_session(self) -> Dict[str, Any]:
        """Compile final session telemetry CSV and session summary JSON at exit (Section 47)."""
        duration = time.time() - self.start_time
        total_frames = len(self.records)

        if total_frames > 0:
            df = pd.DataFrame(self.records)
            df.to_csv(self.csv_path, index=False)
            print(f"[Logger] Session telemetry saved to {self.csv_path}")

        avg_fps = float(np.mean(self.fps_list)) if self.fps_list else 0.0
        min_fps = float(np.min(self.fps_list)) if self.fps_list else 0.0
        max_dvi = float(np.max(self.dvi_list)) if self.dvi_list else 0.0
        avg_dvi = float(np.mean(self.dvi_list)) if self.dvi_list else 0.0

        time_per_state = {
            state: float(count / max(total_frames, 1) * duration)
            for state, count in self.state_counts.items()
        }

        summary = {
            "session_id": self.session_id,
            "session_duration_sec": float(duration),
            "total_frames_processed": total_frames,
            "average_fps": avg_fps,
            "minimum_fps": min_fps,
            "state_distribution_frames": self.state_counts,
            "time_in_state_seconds": time_per_state,
            "maximum_dvi": max_dvi,
            "average_dvi": avg_dvi,
            "blink_count": self.blink_count,
            "yawn_count": self.yawn_count,
            "alarm_events_count": self.alarm_events_count,
            "telemetry_csv": str(self.csv_path)
        }

        with open(self.summary_json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)

        print(f"[Logger] Session summary saved to {self.summary_json_path}")
        return summary
