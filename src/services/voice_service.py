"""
Optional voice synthesis service.

Uses Sarvam TTS when an API key is available and falls back to metadata-only
responses so the UI can use browser speech synthesis locally.
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional
from urllib import request, error


SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"


LANGUAGE_PROFILES = {
    "english": {"language_code": "en-IN", "speaker": "ishita"},
    "hindi": {"language_code": "hi-IN", "speaker": "priya"},
    "hinglish": {"language_code": "en-IN", "speaker": "ishita"},
}


@dataclass
class VoiceSynthesisResult:
    provider: str
    language: str
    language_code: str
    text: str
    speaker: Optional[str] = None
    audio_bytes: Optional[bytes] = None
    mime_type: str = "audio/wav"
    error: Optional[str] = None

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "language": self.language,
            "language_code": self.language_code,
            "text": self.text,
            "speaker": self.speaker,
            "mime_type": self.mime_type,
            "has_audio": self.audio_bytes is not None,
            "error": self.error,
        }


class VoiceSynthesisService:
    """Generate speech audio using Sarvam when configured, else return metadata."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("SARVAM_API_KEY", "").strip()
        self.enabled = bool(self.api_key)

    def profile_for(self, language: str) -> Dict[str, str]:
        return LANGUAGE_PROFILES.get(language.lower(), LANGUAGE_PROFILES["english"])

    def build_prompt(self, language: str, text: str) -> VoiceSynthesisResult:
        profile = self.profile_for(language)
        if not self.enabled:
            return VoiceSynthesisResult(
                provider="browser-fallback",
                language=language,
                language_code=profile["language_code"],
                text=text,
                speaker=profile["speaker"],
            )

        payload = {
            "text": text,
            "model": "bulbul:v3",
            "language_code": profile["language_code"],
            "speaker": profile["speaker"],
            "output_audio_codec": "wav",
        }

        req = request.Request(
            SARVAM_TTS_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "api-subscription-key": self.api_key,
            },
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=20) as response:
                body = response.read().decode("utf-8")
            decoded = json.loads(body)
            audio_items = decoded.get("audios") or []
            if audio_items:
                audio_bytes = base64.b64decode("".join(audio_items))
                return VoiceSynthesisResult(
                    provider="sarvam",
                    language=language,
                    language_code=profile["language_code"],
                    text=text,
                    speaker=profile["speaker"],
                    audio_bytes=audio_bytes,
                    mime_type="audio/wav",
                )
            return VoiceSynthesisResult(
                provider="sarvam",
                language=language,
                language_code=profile["language_code"],
                text=text,
                speaker=profile["speaker"],
                error="Sarvam returned no audio data.",
            )
        except error.HTTPError as exc:
            return VoiceSynthesisResult(
                provider="sarvam",
                language=language,
                language_code=profile["language_code"],
                text=text,
                speaker=profile["speaker"],
                error=f"Sarvam HTTP error {exc.code}",
            )
        except Exception as exc:
            return VoiceSynthesisResult(
                provider="sarvam",
                language=language,
                language_code=profile["language_code"],
                text=text,
                speaker=profile["speaker"],
                error=str(exc),
            )

    def browser_prompt(self, text: str, language: str) -> Dict[str, Any]:
        profile = self.profile_for(language)
        return {
            "provider": "browser-fallback",
            "language": language,
            "language_code": profile["language_code"],
            "text": text,
            "speaker": profile["speaker"],
        }

