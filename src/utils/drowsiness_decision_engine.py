"""
Drowsiness Safety Decision Engine & State Machine.
Implements post-inference temporal safety decision layer:
ALERT -> SUSPECT -> DROWSY -> RECOVERY -> ALERT
Prevents false-positive drowsiness alarms caused by normal eye blinks (<=0.35s) or single noisy frames,
while preserving raw model predictions and enforcing temporal persistence and hysteresis.
"""

from collections import deque
from typing import Dict, Any, Optional
import numpy as np

class DrowsinessDecisionEngine:
    """
    Temporal Safety Decision Engine & State Machine for Driver Safety AI.
    Combines deep learning model predictions with a continuous probability accumulator,
    temporal persistence filter, and hysteresis to suppress transient false positives (e.g. normal blinks).
    
    States:
      - ALERT           : Normal attentive driving / transient blinks (drowsy_duration < 0.4s).
      - SUSPECT         : Suspicious eye closure / elevated probability (0.4s <= drowsy_duration < 0.8s).
      - DROWSY          : Confirmed drowsiness warning (0.8s <= drowsy_duration < 1.2s).
      - HIGH DROWSINESS : Critical prolonged drowsiness; alarm and strobe active (drowsy_duration >= 1.2s).
      - RECOVERY        : Driver eyes reopened; stability recovery check active.
    """
    
    STATE_ALERT = "ALERT"
    STATE_SUSPECT = "SUSPECT"
    STATE_DROWSY = "DROWSY"
    STATE_HIGH_DROWSINESS = "HIGH DROWSINESS"
    STATE_RECOVERY = "RECOVERY"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {}
        
        cfg = config.get("drowsiness", {})
        self.probability_threshold = cfg.get("probability_threshold", 0.70)
        self.exit_threshold = cfg.get("hysteresis_exit_threshold", 0.40)
        self.suspect_duration = cfg.get("suspect_duration_seconds", 0.40)
        self.drowsy_confirmation_duration = cfg.get("confirmation_duration_seconds", 0.80)
        self.high_drowsiness_duration = cfg.get("high_drowsiness_duration_seconds", 1.20)
        self.recovery_duration = cfg.get("recovery_duration_seconds", 1.50)
        self.decay_rate = cfg.get("decay_rate", 1.5) # Accumulator decay factor when P < threshold
        self.blink_max_duration = cfg.get("blink_max_duration_seconds", 0.35)

        # Internal State & Accumulators
        self.state = self.STATE_ALERT
        self.drowsy_timer = 0.0 # Continuous probability accumulator (seconds)
        self.prev_timestamp: Optional[float] = None
        self.recovery_start_time: Optional[float] = None

    def reset(self):
        """Reset decision engine state and accumulators."""
        self.state = self.STATE_ALERT
        self.drowsy_timer = 0.0
        self.prev_timestamp = None
        self.recovery_start_time = None

    def process_frame(
        self,
        p_drowsy: float,
        mean_ear: float,
        eye_closure_duration: float,
        perclos: float,
        is_valid_face: bool = True,
        timestamp_sec: float = 0.0
    ) -> Dict[str, Any]:
        """
        Process single-frame inference probability and telemetry through temporal accumulator & state machine.
        """
        p_drowsy = float(p_drowsy)
        eye_closure_duration = float(eye_closure_duration)
        perclos = float(perclos)

        if self.prev_timestamp is None or timestamp_sec <= self.prev_timestamp:
            dt = 0.033 # Default ~30 FPS frame delta
        else:
            dt = min(timestamp_sec - self.prev_timestamp, 2.0)
        self.prev_timestamp = timestamp_sec

        if not is_valid_face:
            # Face lost -> decay timer slowly
            self.drowsy_timer = max(0.0, self.drowsy_timer - dt)
            return self._build_result(p_drowsy, eye_closure_duration, perclos)

        # 1. Temporal Probability Accumulator + Decay
        if p_drowsy >= self.probability_threshold or eye_closure_duration > self.blink_max_duration:
            self.drowsy_timer += dt
        else:
            # Decay timer when probability drops / eyes open
            self.drowsy_timer = max(0.0, self.drowsy_timer - self.decay_rate * dt)

        # 2. State Machine Transitions & Hysteresis
        if self.state in [self.STATE_DROWSY, self.STATE_HIGH_DROWSINESS]:
            if p_drowsy < self.exit_threshold and eye_closure_duration == 0.0:
                self.state = self.STATE_RECOVERY
                self.recovery_start_time = timestamp_sec

        if self.state in [self.STATE_ALERT, self.STATE_SUSPECT, self.STATE_DROWSY, self.STATE_HIGH_DROWSINESS]:
            if self.drowsy_timer >= self.high_drowsiness_duration:
                self.state = self.STATE_HIGH_DROWSINESS
            elif self.drowsy_timer >= self.drowsy_confirmation_duration:
                self.state = self.STATE_DROWSY
            elif self.drowsy_timer >= self.suspect_duration:
                self.state = self.STATE_SUSPECT
            else:
                self.state = self.STATE_ALERT

        elif self.state == self.STATE_RECOVERY:
            if self.recovery_start_time is None:
                self.recovery_start_time = timestamp_sec

            recovery_elapsed = timestamp_sec - self.recovery_start_time

            # If eyes remain open and P < exit threshold for recovery_duration -> return to ALERT
            if p_drowsy < self.exit_threshold and eye_closure_duration == 0.0:
                if recovery_elapsed >= self.recovery_duration and self.drowsy_timer < self.suspect_duration:
                    self.state = self.STATE_ALERT
                    self.recovery_start_time = None
            elif p_drowsy >= self.probability_threshold or eye_closure_duration >= 0.5:
                # Relapse into DROWSY
                self.state = self.STATE_DROWSY
                self.recovery_start_time = None

        return self._build_result(p_drowsy, eye_closure_duration, perclos)

    def _build_result(
        self,
        p_drowsy: float,
        eye_closure_duration: float,
        perclos: float
    ) -> Dict[str, Any]:
        """Formats decision engine status output dictionary."""
        alarm_active = (self.state == self.STATE_HIGH_DROWSINESS) or (self.state == self.STATE_DROWSY and self.drowsy_timer >= 1.0)
        is_normal_blink = (self.state == self.STATE_ALERT and 0.0 < eye_closure_duration <= self.blink_max_duration)

        # UI Subtext Status Message
        if self.state == self.STATE_ALERT:
            if is_normal_blink:
                subtext = "NORMAL BLINK DETECTED (SAFE)"
            else:
                subtext = f"DROWSY Prob: {p_drowsy*100:.1f}%"
        elif self.state == self.STATE_SUSPECT:
            subtext = f"Eye closure detected | Confirming... {self.drowsy_timer:.2f}s"
        elif self.state == self.STATE_DROWSY:
            subtext = f"Drowsiness confirmed | Duration: {self.drowsy_timer:.2f}s"
        elif self.state == self.STATE_HIGH_DROWSINESS:
            subtext = f"CRITICAL DROWSINESS | Duration: {self.drowsy_timer:.2f}s"
        elif self.state == self.STATE_RECOVERY:
            subtext = "RECOVERING | Stability check..."
        else:
            subtext = f"DROWSY Prob: {p_drowsy*100:.1f}%"

        return {
            "state": self.state,
            "p_drowsy": p_drowsy,
            "drowsy_duration": self.drowsy_timer,
            "eye_closure_duration": eye_closure_duration,
            "perclos": perclos,
            "confirmation_timer": self.drowsy_timer,
            "confirmation_target": self.drowsy_confirmation_duration,
            "alarm_triggered": alarm_active,
            "is_normal_blink": is_normal_blink,
            "status_subtext": subtext
        }
