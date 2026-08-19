"""
Adaptive Driver Drowsiness State Manager.
Converts noisy frame-level neural network predictions into a stable safety state machine
using temporal persistence, hysteresis, driver response monitoring, and post-recovery refractory cooldown.
"""

import time
from collections import deque
from typing import Dict, Any, List, Optional


class StateManager:
    """
    Temporal State Manager for Driver Drowsiness Confirmation.

    States:
        ALERT: Attentive driving baseline.
        SUSPECTED_DROWSY: Transient high probability or short closure detected.
        CONFIRMED_DROWSY: Sustained high probability or closure >= confirmation_duration_sec.
        PERSISTENT_DROWSY: Continued drowsiness after initial voice intervention.
        RECOVERING: Eyes reopened and probability < recovery_threshold; verifying stability.

    Integrates hysteresis, response monitoring, post-recovery cooldown, and event timeline logging.
    """

    STATE_ALERT = "ALERT"
    STATE_SUSPECTED = "SUSPECTED_DROWSY"
    STATE_CONFIRMED = "CONFIRMED_DROWSY"
    STATE_PERSISTENT = "PERSISTENT_DROWSY"
    STATE_RECOVERING = "RECOVERING"

    RESPONSE_NONE = "NONE"
    RESPONSE_WAITING = "WAITING"
    RESPONSE_RECOVERED = "RECOVERED"
    RESPONSE_NOT_RECOVERED = "NOT RECOVERED"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {}

        drowsy_cfg = config.get("drowsiness", {})
        self.p_threshold: float = drowsy_cfg.get("probability_threshold", 0.80)
        self.recovery_p_threshold: float = drowsy_cfg.get("recovery_probability_threshold", 0.45)
        self.confirmation_duration_sec: float = drowsy_cfg.get("confirmation_duration_sec", 1.0)
        self.recovery_duration_sec: float = drowsy_cfg.get("recovery_duration_sec", 1.0)
        self.minimum_closure_duration_sec: float = drowsy_cfg.get("minimum_closure_duration_sec", 1.0)
        self.response_window_sec: float = drowsy_cfg.get("response_window_sec", 3.0)
        self.escalation_delay_sec: float = drowsy_cfg.get("escalation_delay_sec", 5.0)
        self.voice_cooldown_sec: float = drowsy_cfg.get("voice_cooldown_sec", 8.0)
        self.post_recovery_cooldown_sec: float = drowsy_cfg.get("post_recovery_cooldown_sec", 5.0)
        self.blink_max_duration: float = drowsy_cfg.get("blink_max_duration_seconds", 0.35)

        # Internal State Variables
        self.state: str = self.STATE_ALERT
        self.prev_state: str = self.STATE_ALERT
        self.drowsy_timer: float = 0.0
        self.recovery_timer: float = 0.0
        self.state_entry_time: float = 0.0
        self.prev_timestamp: Optional[float] = None
        self.session_start_time: Optional[float] = None

        # Cooldowns and Timing
        self.last_voice_time: float = -999.0
        self.last_recovery_time: float = -999.0
        self.intervention_level: int = 0
        self.response_status: str = self.RESPONSE_NONE
        self.response_window_start: Optional[float] = None
        self.voice_event: Optional[str] = None

        # Prediction History Buffer
        self.prob_history: deque = deque(maxlen=30)
        self.consecutive_high_p: int = 0

        # In-Memory Event Timeline (latest 50 events)
        self.event_timeline: deque = deque(maxlen=50)
        self._add_event(f"00:00.00 {self.STATE_ALERT}")

    def reset(self):
        """Reset state manager state for new session."""
        self.state = self.STATE_ALERT
        self.prev_state = self.STATE_ALERT
        self.drowsy_timer = 0.0
        self.recovery_timer = 0.0
        self.state_entry_time = 0.0
        self.prev_timestamp = None
        self.session_start_time = None
        self.last_voice_time = -999.0
        self.last_recovery_time = -999.0
        self.intervention_level = 0
        self.response_status = self.RESPONSE_NONE
        self.response_window_start = None
        self.voice_event = None
        self.prob_history.clear()
        self.consecutive_high_p = 0
        self.event_timeline.clear()
        self._add_event("00:00.00 ALERT")

    def _format_time(self, timestamp_sec: float) -> str:
        """Format timestamp into MM:SS.ms string."""
        if self.session_start_time is None:
            self.session_start_time = timestamp_sec
        elapsed = max(0.0, timestamp_sec - self.session_start_time)
        mins = int(elapsed // 60)
        secs = elapsed % 60
        return f"{mins:02d}:{secs:05.2f}"

    def _add_event(self, text: str):
        """Append entry to event timeline log."""
        self.event_timeline.append(text)

    def process_frame(
        self,
        p_drowsy: float,
        mean_ear: float,
        eye_closure_duration: float = 0.0,
        perclos: float = 0.0,
        timestamp_sec: Optional[float] = None,
        is_valid_face: bool = True
    ) -> Dict[str, Any]:
        """
        Process single frame prediction and update driver safety state machine.

        Args:
            p_drowsy: Neural network output DROWSY probability [0.0 - 1.0].
            mean_ear: Current mean Eye Aspect Ratio.
            eye_closure_duration: Active eye closure duration in seconds.
            perclos: Current PERCLOS metric [0.0 - 1.0].
            timestamp_sec: System or video timestamp in seconds.
            is_valid_face: True if MediaPipe extracted a face landmark.

        Returns:
            Dict containing state info, intervention level, voice event, response status, and event timeline.
        """
        now = time.time() if timestamp_sec is None else timestamp_sec
        if self.session_start_time is None:
            self.session_start_time = now

        # Delta time calculation
        if self.prev_timestamp is None or now <= self.prev_timestamp:
            dt = 0.033
        else:
            dt = min(now - self.prev_timestamp, 2.0)
        self.prev_timestamp = now

        self.voice_event = None

        if not is_valid_face:
            # Decay timer when face is lost
            self.drowsy_timer = max(0.0, self.drowsy_timer - 1.5 * dt)
            return self._build_result(p_drowsy, mean_ear, eye_closure_duration, perclos, now)

        # 1. Update rolling prediction history & blink detection
        self.prob_history.append(float(p_drowsy))
        is_normal_blink = (0.0 < eye_closure_duration <= self.blink_max_duration)

        if p_drowsy >= self.p_threshold:
            self.consecutive_high_p += 1
        else:
            self.consecutive_high_p = 0

        # Check Post-Recovery Refractory Cooldown
        in_post_recovery_cooldown = (now - self.last_recovery_time) < self.post_recovery_cooldown_sec

        # 2. Probability & Closure Accumulator
        # Note: Normal short blinks (<=0.35s) do NOT increment drowsiness timer unless sustained
        is_drowsy_signal = (p_drowsy >= self.p_threshold) or (eye_closure_duration > self.blink_max_duration)

        if is_drowsy_signal:
            # During post-recovery cooldown, require stronger confirmation (rate = 0.7x)
            rate = 0.7 if in_post_recovery_cooldown else 1.0
            self.drowsy_timer += dt * rate
        else:
            # Decay rate when evidence drops
            self.drowsy_timer = max(0.0, self.drowsy_timer - 1.5 * dt)

        # 3. State Machine Transitions & Hysteresis
        current_time_in_state = now - self.state_entry_time

        if self.state == self.STATE_ALERT:
            self.intervention_level = 0
            self.response_status = self.RESPONSE_NONE

            if self.drowsy_timer >= 0.35 and not in_post_recovery_cooldown:
                self._transition_to(self.STATE_SUSPECTED, now)
                # Level 1 voice prompt
                if (now - self.last_voice_time) >= self.voice_cooldown_sec or self.intervention_level > 0:
                    self.voice_event = "VOICE_LEVEL_1"
                    self.last_voice_time = now
                    self._add_event(f"{self._format_time(now)} VOICE_LEVEL_1: Please stay alert.")

        elif self.state == self.STATE_SUSPECTED:
            self.intervention_level = 1
            if self.drowsy_timer >= self.confirmation_duration_sec or eye_closure_duration >= self.minimum_closure_duration_sec:
                self._transition_to(self.STATE_CONFIRMED, now)
                self.intervention_level = 2
                self.response_status = self.RESPONSE_WAITING
                self.response_window_start = now
                # Escalation to Level 2 always triggers voice alert
                self.voice_event = "VOICE_LEVEL_2"
                self.last_voice_time = now
                self._add_event(f"{self._format_time(now)} VOICE_LEVEL_2: You appear drowsy. Please stay alert.")
            elif self.drowsy_timer <= 0.0:
                self._transition_to(self.STATE_ALERT, now)

        elif self.state == self.STATE_CONFIRMED:
            self.intervention_level = 2
            # Driver Response Monitoring Window (~3.0s)
            if self.response_window_start is not None:
                elapsed_response = now - self.response_window_start
                if p_drowsy < self.recovery_p_threshold and eye_closure_duration == 0.0:
                    self.response_status = self.RESPONSE_RECOVERED
                    self._transition_to(self.STATE_RECOVERING, now)
                elif elapsed_response >= self.response_window_sec and current_time_in_state >= self.escalation_delay_sec:
                    self.response_status = self.RESPONSE_NOT_RECOVERED
                    self._transition_to(self.STATE_PERSISTENT, now)
                    self.intervention_level = 3
                    # Escalation to Level 3 always triggers voice alert
                    self.voice_event = "VOICE_LEVEL_3"
                    self.last_voice_time = now
                    self._add_event(f"{self._format_time(now)} VOICE_LEVEL_3: Warning. Drowsiness continuing.")

        elif self.state == self.STATE_PERSISTENT:
            self.intervention_level = 3
            if p_drowsy < self.recovery_p_threshold and eye_closure_duration == 0.0:
                self.response_status = self.RESPONSE_RECOVERED
                self._transition_to(self.STATE_RECOVERING, now)

        elif self.state == self.STATE_RECOVERING:
            if p_drowsy < self.recovery_p_threshold and eye_closure_duration == 0.0:
                self.recovery_timer += dt
                if self.recovery_timer >= self.recovery_duration_sec:
                    self.voice_event = "ALERTNESS_RESTORED"
                    self.last_recovery_time = now
                    self.last_voice_time = now
                    self._add_event(f"{self._format_time(now)} RECOVERED: Alertness restored.")
                    self._transition_to(self.STATE_ALERT, now)
            else:
                # Interrupted recovery - return to confirmed/persistent
                self.recovery_timer = max(0.0, self.recovery_timer - 2.0 * dt)
                if p_drowsy >= self.p_threshold:
                    self._transition_to(self.STATE_CONFIRMED, now)

        return self._build_result(p_drowsy, mean_ear, eye_closure_duration, perclos, now, is_normal_blink)

    def _transition_to(self, new_state: str, now: float):
        """Execute state transition and log to event timeline."""
        if self.state != new_state:
            self.prev_state = self.state
            self.state = new_state
            self.state_entry_time = now
            if new_state == self.STATE_RECOVERING:
                self.recovery_timer = 0.0
            elif new_state == self.STATE_ALERT:
                self.drowsy_timer = 0.0
                self.recovery_timer = 0.0

            formatted_time = self._format_time(now)
            self._add_event(f"{formatted_time} {new_state}")

    def _build_result(
        self,
        p_drowsy: float,
        mean_ear: float,
        eye_closure_duration: float,
        perclos: float,
        now: float,
        is_normal_blink: bool = False
    ) -> Dict[str, Any]:
        """Construct standard telemetry dictionary output for HUD and Logger."""
        alarm_active = (self.state in [self.STATE_CONFIRMED, self.STATE_PERSISTENT])

        return {
            "state": self.state,
            "prev_state": self.prev_state,
            "p_drowsy": float(p_drowsy),
            "mean_ear": float(mean_ear),
            "eye_closure_duration": float(eye_closure_duration),
            "perclos": float(perclos),
            "alarm_triggered": alarm_active,
            "intervention_level": self.intervention_level,
            "voice_event": self.voice_event,
            "response_status": self.response_status,
            "is_normal_blink": is_normal_blink,
            "drowsy_timer": round(self.drowsy_timer, 2),
            "event_timeline": list(self.event_timeline)
        }
