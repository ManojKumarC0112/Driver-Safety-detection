"""
Automated Unit Tests for VoiceAlertEngine.
Validates:
1. Voice intervention levels play correct messages.
2. Cooldown timer enforces rate limiting (no frame-by-frame spam).
3. Audio playback failure falls back safely without crashing application.
"""

import time
import pytest
from src.inference.voice_alert import VoiceAlertEngine


@pytest.fixture
def voice_engine():
    config = {
        "drowsiness": {
            "enable_voice": True,
            "enable_fallback_alarm": True,
            "voice_cooldown_sec": 8.0
        }
    }
    return VoiceAlertEngine(config=config)


def test_voice_alert_messages():
    """Verify voice messages dictionary keys."""
    assert "VOICE_LEVEL_1" in VoiceAlertEngine.MESSAGES
    assert "VOICE_LEVEL_2" in VoiceAlertEngine.MESSAGES
    assert "VOICE_LEVEL_3" in VoiceAlertEngine.MESSAGES
    assert "ALERTNESS_RESTORED" in VoiceAlertEngine.MESSAGES
    assert VoiceAlertEngine.MESSAGES["VOICE_LEVEL_1"] == "Please stay alert."


def test_cooldown_enforcement(voice_engine):
    """Cooldown prevents consecutive voice calls within voice_cooldown_sec."""
    # First call succeeds
    res1 = voice_engine.speak("VOICE_LEVEL_1")
    assert res1 is True

    # Immediate second call fails due to 8.0s cooldown
    res2 = voice_engine.speak("VOICE_LEVEL_1")
    assert res2 is False


def test_force_bypasses_cooldown(voice_engine):
    """Setting force=True bypasses cooldown constraint."""
    voice_engine.speak("VOICE_LEVEL_1")
    res_forced = voice_engine.speak("VOICE_LEVEL_2", force=True)
    assert res_forced is True


def test_audio_failure_fallback_does_not_crash():
    """Simulate audio device failure; app must remain functional without crashing."""
    config = {
        "drowsiness": {
            "enable_voice": True,
            "enable_fallback_alarm": True,
            "voice_cooldown_sec": 0.0
        }
    }
    engine = VoiceAlertEngine(config=config)
    # Simulate TTS failure
    engine.audio_failed = True
    engine.engine = None

    # Call speak; must return True (fallback path) without throwing exception
    try:
        res = engine.speak("VOICE_LEVEL_2", force=True)
        assert res is True
    except Exception as e:
        pytest.fail(f"Audio failure crashed application: {e}")
