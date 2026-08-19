"""
Offline Multi-Stage Voice Intervention Engine.
Provides non-blocking voice announcements using Windows SAPI5 / pyttsx3 or winsound fallback.
Supports audio failure handling, visual HUD fallback warnings, queue-based thread execution, and cooldown rate-limiting.
"""

import sys
import time
import queue
import logging
import threading
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

try:
    import win32com.client
    WIN32COM_AVAILABLE = True
except ImportError:
    WIN32COM_AVAILABLE = False

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

try:
    import winsound
    WINSOUND_AVAILABLE = True
except ImportError:
    WINSOUND_AVAILABLE = False

try:
    import pythoncom
    PYTHONCOM_AVAILABLE = True
except ImportError:
    PYTHONCOM_AVAILABLE = False


class VoiceAlertEngine:
    """
    Queue-based multi-stage offline voice intervention system.
    Runs a single background thread that initializes SAPI5 COM context to ensure 100% reliable voice synthesis.

    Levels:
        Level 0: Silent
        Level 1: "Please stay alert."
        Level 2: "You appear drowsy. Please stay alert."
        Level 3: "Warning. Drowsiness is continuing. Please take a safe break."
        Recovery: "Alertness restored."
    """

    MESSAGES = {
        "VOICE_LEVEL_1": "Please stay alert.",
        "VOICE_LEVEL_2": "You appear drowsy. Please stay alert.",
        "VOICE_LEVEL_3": "Warning. Drowsiness is continuing. Please take a safe break.",
        "ALERTNESS_RESTORED": "Alertness restored."
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {}

        drowsy_cfg = config.get("drowsiness", {})
        self.enable_voice: bool = drowsy_cfg.get("enable_voice", True)
        self.enable_fallback_alarm: bool = drowsy_cfg.get("enable_fallback_alarm", True)
        self.voice_cooldown_sec: float = drowsy_cfg.get("voice_cooldown_sec", 8.0)

        self.last_speech_time: float = -999.0
        self.last_event_key: Optional[str] = None
        self.audio_failed: bool = False
        self.last_audio_error: Optional[str] = None
        self.is_speaking: bool = False

        # Queue for non-blocking worker thread
        self.msg_queue: queue.Queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def speak(self, event_key: str, force: bool = False) -> bool:
        """
        Speak specified voice intervention event non-blockingly.

        Args:
            event_key: Key in MESSAGES dict ('VOICE_LEVEL_1', 'VOICE_LEVEL_2', 'VOICE_LEVEL_3', 'ALERTNESS_RESTORED').
            force: Bypass cooldown check if True.

        Returns:
            True if audio message was queued; False otherwise.
        """
        if not self.enable_voice or event_key not in self.MESSAGES:
            return False

        now = time.time()
        is_new_event = (event_key != self.last_event_key)
        if not force and not is_new_event and (now - self.last_speech_time) < self.voice_cooldown_sec:
            return False

        message_text = self.MESSAGES[event_key]
        self.last_speech_time = now
        self.last_event_key = event_key

        # Add to non-blocking queue (drop extra if queue backed up to prevent delay)
        if self.msg_queue.qsize() < 4:
            self.msg_queue.put((event_key, message_text))
            return True
        return False

    def _worker_loop(self):
        """Single background worker thread handling COM initialization and TTS playback."""
        # Initialize COM context for thread on Windows
        if PYTHONCOM_AVAILABLE:
            try:
                pythoncom.CoInitialize()
            except Exception:
                pass

        sapi_voice = None
        if self.enable_voice and WIN32COM_AVAILABLE and sys.platform == "win32":
            try:
                sapi_voice = win32com.client.Dispatch("SAPI.SpVoice")
            except Exception as e:
                logger.warning(f"Could not initialize native SAPI.SpVoice: {e}")

        engine = None
        if sapi_voice is None and self.enable_voice and PYTTSX3_AVAILABLE:
            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", 155)
                engine.setProperty("volume", 1.0)
            except Exception as e:
                logger.warning(f"Failed to initialize pyttsx3 engine: {e}")
                self.audio_failed = True
                self.last_audio_error = str(e)

        while True:
            try:
                event_key, message_text = self.msg_queue.get()

                self.is_speaking = True

                if sapi_voice is not None:
                    try:
                        sapi_voice.Speak(message_text, 1)  # 1 = SVSFlagsAsync
                    except Exception as e:
                        logger.error(f"[SAPI AUDIO ERROR] Native SAPI playback failed: {e}")
                        sapi_voice = None
                        if engine is None and PYTTSX3_AVAILABLE:
                            try:
                                engine = pyttsx3.init()
                                engine.setProperty("rate", 155)
                            except Exception:
                                pass

                if sapi_voice is None and engine is not None and not self.audio_failed:
                    try:
                        engine.say(message_text)
                        engine.runAndWait()
                    except Exception as e:
                        logger.error(f"[AUDIO ERROR] TTS playback failed: {e}")
                        self.audio_failed = True
                        self.last_audio_error = str(e)
                        if WINSOUND_AVAILABLE and self.enable_fallback_alarm:
                            try:
                                winsound.Beep(1000, 300)
                            except Exception:
                                pass
                elif sapi_voice is None and (engine is None or self.audio_failed) and WINSOUND_AVAILABLE and self.enable_fallback_alarm:
                    freq = 1200 if "LEVEL_3" in event_key else 800
                    dur = 400 if "LEVEL_3" in event_key else 200
                    winsound.Beep(freq, dur)

                self.is_speaking = False
                self.msg_queue.task_done()
            except Exception as e:
                logger.error(f"Voice worker exception: {e}")
                time.sleep(0.1)

    def trigger_fallback_beep(self):
        """Direct trigger for hardware fallback beep."""
        if WINSOUND_AVAILABLE:
            try:
                winsound.Beep(1000, 300)
            except Exception:
                pass
