"""Optional TTS export for podcast scripts (gTTS with graceful fallback)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src import config
from src.utils import logger


class TTSService:
    def __init__(self, output_dir: Path | None = None) -> None:
        self.output_dir = output_dir or (config.STORAGE_DIR / "podcasts")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_podcast(self, document_id: str, script: dict[str, Any]) -> dict[str, Any]:
        turns = script.get("turns") or []
        text = "\n".join(f"{t.get('speaker', 'Host')}: {t.get('text', '')}" for t in turns)
        if not text.strip():
            return {"status": "empty", "audio_uri": None}

        filename = f"{document_id}_podcast.mp3"
        target = self.output_dir / filename

        try:
            from gtts import gTTS

            gTTS(text=text[:5000], lang="vi").save(str(target))
            provider = "gtts"
        except Exception as exc:
            logger.warning("gTTS unavailable (%s); writing text stub only", exc)
            stub = self.output_dir / f"{document_id}_podcast.txt"
            stub.write_text(text, encoding="utf-8")
            return {
                "status": "text_only",
                "audio_uri": str(stub),
                "tts_provider": "stub",
                "message": "Install gTTS for MP3 export: pip install gtts",
            }

        return {
            "status": "ready",
            "audio_uri": str(target),
            "tts_provider": provider,
            "bytes": target.stat().st_size if target.exists() else 0,
        }
