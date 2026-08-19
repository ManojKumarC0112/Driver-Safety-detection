"""
Unit Tests for DrowsinessDecisionEngine State Machine & Temporal Safety Layer.
Validates:
- Normal blink (<=0.35s) does not trigger DROWSY or alarm
- Continuous probability accumulator + decay rate
- 4-stage state machine (ALERT -> SUSPECT -> DROWSY -> HIGH DROWSINESS)
- Single probability spike does not immediately trigger DROWSY
- Sustained probability triggers DROWSY and HIGH DROWSINESS alarm
- Recovery window returns state to ALERT
- Hysteresis prevents state oscillation near threshold
"""

import pytest
from src.utils.drowsiness_decision_engine import DrowsinessDecisionEngine

@pytest.fixture
def decision_engine():
    config = {
        "drowsiness": {
            "probability_threshold": 0.70,
            "hysteresis_exit_threshold": 0.40,
            "suspect_duration_seconds": 0.40,
            "confirmation_duration_seconds": 0.80,
            "high_drowsiness_duration_seconds": 1.20,
            "recovery_duration_seconds": 1.50,
            "decay_rate": 1.5,
            "blink_max_duration_seconds": 0.35
        }
    }
    return DrowsinessDecisionEngine(config=config)

def test_normal_blink_does_not_trigger_drowsy(decision_engine):
    """Normal short blink (0.2s closure) must remain ALERT without alarm."""
    res1 = decision_engine.process_frame(p_drowsy=0.05, mean_ear=0.45, eye_closure_duration=0.0, perclos=0.0, timestamp_sec=0.0)
    assert res1["state"] == "ALERT"
    assert not res1["alarm_triggered"]

    # Short blink (0.2s)
    res2 = decision_engine.process_frame(p_drowsy=0.40, mean_ear=0.15, eye_closure_duration=0.20, perclos=0.05, timestamp_sec=0.2)
    assert res2["state"] == "ALERT"
    assert res2["is_normal_blink"]
    assert not res2["alarm_triggered"]

    # Eyes reopened
    res3 = decision_engine.process_frame(p_drowsy=0.05, mean_ear=0.45, eye_closure_duration=0.0, perclos=0.0, timestamp_sec=0.3)
    assert res3["state"] == "ALERT"
    assert not res3["alarm_triggered"]

def test_prolonged_closure_triggers_drowsy_and_high_drowsiness(decision_engine):
    """Prolonged closure progresses ALERT -> SUSPECT -> DROWSY -> HIGH DROWSINESS."""
    # Start t=0.0
    decision_engine.process_frame(p_drowsy=0.05, mean_ear=0.45, eye_closure_duration=0.0, perclos=0.0, timestamp_sec=0.0)

    # t=0.2s (blink) -> ALERT
    res_blink = decision_engine.process_frame(p_drowsy=0.85, mean_ear=0.15, eye_closure_duration=0.2, perclos=0.05, timestamp_sec=0.2)
    assert res_blink["state"] == "ALERT"

    # t=0.5s (accumulator = 0.5s >= 0.4s) -> SUSPECT
    res_suspect = decision_engine.process_frame(p_drowsy=0.85, mean_ear=0.15, eye_closure_duration=0.5, perclos=0.2, timestamp_sec=0.5)
    assert res_suspect["state"] == "SUSPECT"

    # t=0.9s (accumulator = 0.9s >= 0.8s) -> DROWSY
    res_drowsy = decision_engine.process_frame(p_drowsy=0.85, mean_ear=0.10, eye_closure_duration=0.9, perclos=0.4, timestamp_sec=0.9)
    assert res_drowsy["state"] == "DROWSY"

    # t=1.3s (accumulator = 1.3s >= 1.2s) -> HIGH DROWSINESS
    res_high = decision_engine.process_frame(p_drowsy=0.90, mean_ear=0.10, eye_closure_duration=1.3, perclos=0.6, timestamp_sec=1.3)
    assert res_high["state"] == "HIGH DROWSINESS"
    assert res_high["alarm_triggered"]

def test_single_probability_spike_does_not_immediately_trigger_drowsy(decision_engine):
    """Single frame high probability spike must not jump straight to DROWSY or trigger alarm."""
    res1 = decision_engine.process_frame(p_drowsy=0.95, mean_ear=0.40, eye_closure_duration=0.0, perclos=0.0, timestamp_sec=0.0)
    assert res1["state"] == "ALERT"
    assert not res1["alarm_triggered"]

def test_recovery_returns_to_alert(decision_engine):
    """After eyes reopen, system enters RECOVERY and returns to ALERT after recovery_duration."""
    # Force into DROWSY
    decision_engine.process_frame(p_drowsy=0.85, mean_ear=0.15, eye_closure_duration=0.0, perclos=0.0, timestamp_sec=0.0)
    decision_engine.process_frame(p_drowsy=0.85, mean_ear=0.15, eye_closure_duration=0.9, perclos=0.4, timestamp_sec=0.9)

    # Driver reopens eyes with low probability
    res_rec = decision_engine.process_frame(p_drowsy=0.10, mean_ear=0.45, eye_closure_duration=0.0, perclos=0.0, timestamp_sec=1.0)
    assert res_rec["state"] == "RECOVERY"

    # Maintain open eyes for recovery_duration (1.5s) -> t = 1.0 + 1.6 = 2.6s
    # Accumulator decays, recovery timer completes
    res_alert = decision_engine.process_frame(p_drowsy=0.05, mean_ear=0.45, eye_closure_duration=0.0, perclos=0.0, timestamp_sec=2.6)
    assert res_alert["state"] == "ALERT"

def test_hysteresis_prevents_state_oscillation(decision_engine):
    """Drowsy state should not oscillate when probability fluctuates near entering threshold."""
    # Reach DROWSY state
    decision_engine.process_frame(p_drowsy=0.85, mean_ear=0.15, eye_closure_duration=0.0, perclos=0.0, timestamp_sec=0.0)
    decision_engine.process_frame(p_drowsy=0.85, mean_ear=0.15, eye_closure_duration=0.9, perclos=0.4, timestamp_sec=0.9)

    # Fluctuate probability slightly below 0.70 (e.g. 0.60), but eyes remain closed (>0.35s)
    res_fluct = decision_engine.process_frame(p_drowsy=0.60, mean_ear=0.18, eye_closure_duration=1.0, perclos=0.3, timestamp_sec=1.0)
    assert res_fluct["state"] in ["DROWSY", "HIGH DROWSINESS"]

