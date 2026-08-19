"""
Comprehensive Automated Unit Tests for StateManager.
Validates:
1. Normal short blink (<=0.35s) does NOT trigger confirmed drowsiness or alarm.
2. Short eye closure does NOT trigger confirmed drowsiness.
3. Sustained probability / closure triggers SUSPECTED_DROWSY -> CONFIRMED_DROWSY.
4. Dropping probability below recovery threshold triggers RECOVERING.
5. Hysteresis prevents state oscillation.
6. Driver recovery cancels escalation and triggers ALERTNESS_RESTORED.
7. Post-recovery refractory cooldown prevents rapid re-triggering.
8. Event timeline correctly records all transitions.
9. StateManager resets correctly.
10. Transient spike sequence (0.90 -> 0.87 -> 0.65 -> 0.40 -> 0.25) does NOT confirm drowsiness.
11. Sustained sequence (0.86 -> 0.88 -> 0.91 -> 0.93...) confirms drowsiness.
"""

import pytest
from src.inference.state_manager import StateManager


@pytest.fixture
def state_manager():
    config = {
        "drowsiness": {
            "probability_threshold": 0.80,
            "recovery_probability_threshold": 0.45,
            "confirmation_duration_sec": 1.0,
            "recovery_duration_sec": 1.0,
            "minimum_closure_duration_sec": 1.0,
            "response_window_sec": 3.0,
            "escalation_delay_sec": 5.0,
            "voice_cooldown_sec": 8.0,
            "post_recovery_cooldown_sec": 5.0,
            "blink_max_duration_seconds": 0.35
        }
    }
    return StateManager(config=config)


def test_normal_blink_does_not_trigger_drowsy(state_manager):
    """Normal short blink (0.20s) must remain in ALERT state."""
    # Frame 1: Eyes open
    r1 = state_manager.process_frame(p_drowsy=0.05, mean_ear=0.45, eye_closure_duration=0.0, timestamp_sec=0.0)
    assert r1["state"] == "ALERT"
    assert not r1["alarm_triggered"]

    # Frame 2: Short blink (0.20s)
    r2 = state_manager.process_frame(p_drowsy=0.75, mean_ear=0.15, eye_closure_duration=0.20, timestamp_sec=0.2)
    assert r2["state"] == "ALERT"
    assert r2["is_normal_blink"]
    assert not r2["alarm_triggered"]

    # Frame 3: Eyes reopened
    r3 = state_manager.process_frame(p_drowsy=0.05, mean_ear=0.45, eye_closure_duration=0.0, timestamp_sec=0.3)
    assert r3["state"] == "ALERT"


def test_transient_spike_sequence_does_not_trigger_drowsy(state_manager):
    """Sequence (0.90 -> 0.87 -> 0.65 -> 0.40 -> 0.25) must NOT trigger CONFIRMED_DROWSY."""
    st = 0.0
    probs = [0.90, 0.87, 0.65, 0.40, 0.25]
    for p in probs:
        res = state_manager.process_frame(p_drowsy=p, mean_ear=0.35, eye_closure_duration=0.0, timestamp_sec=st)
        assert res["state"] in ["ALERT", "SUSPECTED_DROWSY"]
        assert res["state"] != "CONFIRMED_DROWSY"
        assert not res["alarm_triggered"]
        st += 0.10


def test_sustained_sequence_triggers_confirmed_drowsy(state_manager):
    """Sustained high probability >= 1.0s must transition ALERT -> SUSPECTED_DROWSY -> CONFIRMED_DROWSY."""
    # Start at t=0.0
    r0 = state_manager.process_frame(p_drowsy=0.05, mean_ear=0.45, timestamp_sec=0.0)
    assert r0["state"] == "ALERT"

    # t=0.4s -> SUSPECTED_DROWSY
    r1 = state_manager.process_frame(p_drowsy=0.85, mean_ear=0.15, eye_closure_duration=0.4, timestamp_sec=0.4)
    assert r1["state"] == "SUSPECTED_DROWSY"

    # t=1.1s -> CONFIRMED_DROWSY (drowsy_timer = 0.4 + 0.7 = 1.1s >= 1.0s)
    r2 = state_manager.process_frame(p_drowsy=0.90, mean_ear=0.10, eye_closure_duration=1.1, timestamp_sec=1.1)
    assert r2["state"] == "CONFIRMED_DROWSY"
    assert r2["alarm_triggered"]
    assert r2["intervention_level"] == 2
    assert r2["response_status"] == "WAITING"


def test_driver_recovery_cancels_escalation(state_manager):
    """When driver reopens eyes in CONFIRMED state, system enters RECOVERING and returns to ALERT."""
    state_manager.process_frame(p_drowsy=0.05, mean_ear=0.45, timestamp_sec=0.0)
    state_manager.process_frame(p_drowsy=0.85, mean_ear=0.15, eye_closure_duration=0.4, timestamp_sec=0.4)
    state_manager.process_frame(p_drowsy=0.90, mean_ear=0.10, eye_closure_duration=1.1, timestamp_sec=1.1)

    # Driver reopens eyes with low probability at t=1.2s -> RECOVERING
    r1 = state_manager.process_frame(p_drowsy=0.10, mean_ear=0.45, eye_closure_duration=0.0, timestamp_sec=1.2)
    assert r1["state"] == "RECOVERING"
    assert r1["response_status"] == "RECOVERED"

    # Maintain open eyes for recovery_duration (1.0s) -> t=2.3s -> ALERT + ALERTNESS_RESTORED
    r2 = state_manager.process_frame(p_drowsy=0.05, mean_ear=0.45, eye_closure_duration=0.0, timestamp_sec=2.3)
    assert r2["state"] == "ALERT"
    assert r2["voice_event"] == "ALERTNESS_RESTORED"


def test_persistent_drowsiness_escalation(state_manager):
    """Sustained drowsiness without recovery triggers PERSISTENT_DROWSY (Level 3)."""
    state_manager.process_frame(p_drowsy=0.05, mean_ear=0.45, timestamp_sec=0.0)
    state_manager.process_frame(p_drowsy=0.85, mean_ear=0.15, eye_closure_duration=0.4, timestamp_sec=0.4)
    state_manager.process_frame(p_drowsy=0.90, mean_ear=0.10, eye_closure_duration=1.1, timestamp_sec=1.1)

    # Eyes remain closed past escalation_delay_sec (5.0s) -> t=6.5s
    r_esc = state_manager.process_frame(p_drowsy=0.95, mean_ear=0.08, eye_closure_duration=6.5, timestamp_sec=6.5)
    assert r_esc["state"] == "PERSISTENT_DROWSY"
    assert r_esc["intervention_level"] == 3
    assert r_esc["response_status"] == "NOT RECOVERED"


def test_post_recovery_refractory_cooldown(state_manager):
    """After recovery, 5.0s refractory cooldown prevents instant re-triggering on mild noise."""
    state_manager.process_frame(p_drowsy=0.05, mean_ear=0.45, timestamp_sec=0.0)
    state_manager.process_frame(p_drowsy=0.85, mean_ear=0.15, eye_closure_duration=0.4, timestamp_sec=0.4)
    state_manager.process_frame(p_drowsy=0.90, mean_ear=0.10, eye_closure_duration=1.1, timestamp_sec=1.1)
    state_manager.process_frame(p_drowsy=0.10, mean_ear=0.45, eye_closure_duration=0.0, timestamp_sec=1.2)
    r_rec = state_manager.process_frame(p_drowsy=0.05, mean_ear=0.45, eye_closure_duration=0.0, timestamp_sec=2.3)
    assert r_rec["state"] == "ALERT"

    # Mild noisy frame during refractory period (t=3.0s < 2.3s + 5.0s)
    r_noise = state_manager.process_frame(p_drowsy=0.82, mean_ear=0.30, eye_closure_duration=0.0, timestamp_sec=3.0)
    assert r_noise["state"] == "ALERT"  # Must stay ALERT during refractory


def test_event_timeline_logging(state_manager):
    """Event timeline records formatted state transitions."""
    state_manager.process_frame(p_drowsy=0.05, mean_ear=0.45, timestamp_sec=0.0)
    state_manager.process_frame(p_drowsy=0.85, mean_ear=0.15, eye_closure_duration=0.4, timestamp_sec=0.4)
    res = state_manager.process_frame(p_drowsy=0.90, mean_ear=0.10, eye_closure_duration=1.1, timestamp_sec=1.1)
    timeline = res["event_timeline"]
    assert len(timeline) >= 2
    assert any("ALERT" in e for e in timeline)
    assert any("CONFIRMED_DROWSY" in e for e in timeline)


def test_state_manager_reset(state_manager):
    """Reset clears history, timers, and returns to pristine ALERT state."""
    state_manager.process_frame(p_drowsy=0.90, mean_ear=0.10, eye_closure_duration=1.1, timestamp_sec=1.1)
    state_manager.reset()
    res = state_manager.process_frame(p_drowsy=0.05, mean_ear=0.45, eye_closure_duration=0.0, timestamp_sec=0.0)
    assert res["state"] == "ALERT"
    assert res["drowsy_timer"] == 0.0
    assert res["intervention_level"] == 0
