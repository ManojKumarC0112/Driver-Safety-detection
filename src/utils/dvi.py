"""
Driver Vigilance Index (DVI) Calculation Engine.
Calculates a 0-100 normalized risk score based on deep learning model probabilities, PERCLOS,
eye closure duration, and head pose deviation.

DISCLAIMER: DVI is a project-defined academic risk index and is not a certified automotive safety metric.
"""

from typing import Tuple, Dict, Any

class DVIEngine:
    """
    Computes Driver Vigilance Index (DVI) score (0.0 to 100.0) and assigns risk levels:
      0 - 25:  LOW
      25 - 50: MODERATE
      50 - 75: HIGH
      75 - 100: CRITICAL
    """
    def __init__(
        self,
        weight_p_drowsy: float = 0.40,
        weight_perclos: float = 0.30,
        weight_eye_closure: float = 0.20,
        weight_yaw_dev: float = 0.10,
        max_eye_closure_sec: float = 3.0,
        max_yaw_dev_deg: float = 45.0,
        low_threshold: float = 25.0,
        mod_threshold: float = 50.0,
        high_threshold: float = 75.0,
    ):
        self.w_p = weight_p_drowsy
        self.w_perclos = weight_perclos
        self.w_eye = weight_eye_closure
        self.w_yaw = weight_yaw_dev

        self.max_eye_closure_sec = max_eye_closure_sec
        self.max_yaw_dev_deg = max_yaw_dev_deg

        self.low_thresh = low_threshold
        self.mod_thresh = mod_threshold
        self.high_thresh = high_threshold

    def calculate_dvi(
        self,
        p_alert: float,
        perclos: float,
        eye_closure_duration_sec: float,
        yaw_deg: float
    ) -> Tuple[float, str, Dict[str, float]]:
        """
        Calculate DVI score and risk level.
        Inputs:
            p_alert (float): Model probability for ALERT class (0.0 to 1.0)
            perclos (float): PERCLOS metric (0.0 to 1.0)
            eye_closure_duration_sec (float): Consecutive closed eye duration in seconds
            yaw_deg (float): Head yaw angle in degrees
        Returns:
            dvi_score (float): Clamped 0.0 to 100.0
            risk_level (str): LOW, MODERATE, HIGH, CRITICAL
            breakdown (dict): Normalized component scores
        """
        # Component 1: Drowsiness probability component (1 - P_ALERT)
        comp_drowsy = float(1.0 - max(0.0, min(1.0, p_alert)))

        # Component 2: PERCLOS (already 0.0 to 1.0)
        comp_perclos = float(max(0.0, min(1.0, perclos)))

        # Component 3: Normalized eye closure duration
        comp_eye_closure = float(min(1.0, max(0.0, eye_closure_duration_sec / self.max_eye_closure_sec)))

        # Component 4: Normalized yaw deviation
        comp_yaw_dev = float(min(1.0, max(0.0, abs(yaw_deg) / self.max_yaw_dev_deg)))

        # Weighted combination (0.0 to 1.0)
        raw_dvi = (
            self.w_p * comp_drowsy +
            self.w_perclos * comp_perclos +
            self.w_eye * comp_eye_closure +
            self.w_yaw * comp_yaw_dev
        )

        # Scale to 0 - 100 and clamp
        dvi_score = float(max(0.0, min(100.0, raw_dvi * 100.0)))

        # Risk level determination
        if dvi_score < self.low_thresh:
            risk_level = "LOW"
        elif dvi_score < self.mod_thresh:
            risk_level = "MODERATE"
        elif dvi_score < self.high_thresh:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"

        breakdown = {
            "comp_drowsy": comp_drowsy,
            "comp_perclos": comp_perclos,
            "comp_eye_closure": comp_eye_closure,
            "comp_yaw_dev": comp_yaw_dev,
            "raw_dvi": raw_dvi
        }

        return dvi_score, risk_level, breakdown
