"""
SceneFlow API — Configuration
Loads environment variables and provides a typed Settings singleton.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ──────────────────────────────────────────────
# Load .env from project root (two levels up from this file)
# ──────────────────────────────────────────────
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_env_path)


class Settings:
    """Application-wide configuration pulled from environment variables."""

    # ── Gemini ──────────────────────────────────
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    # ── Redis / Celery ──────────────────────────
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # ── TTS defaults ────────────────────────────
    TTS_MODEL: str = os.getenv("TTS_MODEL", "edge-tts")
    TTS_VOICE: str = os.getenv("TTS_VOICE", "en-US-AriaNeural")

    # ── Output directory for rendered videos ────
    OUTPUT_DIR: Path = Path(os.getenv("OUTPUT_DIR", "./output")).resolve()

    def validate(self) -> None:
        """Raise early if critical config is missing."""
        if not self.GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY is not set. "
                "Copy .env.example to .env and add your key."
            )


# Singleton — import this everywhere
settings = Settings()
