from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# Optionally load a .env file next to this module (the README documents it).
# python-dotenv is a soft dependency; config still works without it.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:  # pragma: no cover - dotenv is optional
    pass


@dataclass
class GestureConfig:
    # Project paths
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent.parent
    DATASET_DIR: Path = PROJECT_ROOT / "dataset"
    MODELS_DIR: Path = PROJECT_ROOT / "models"

    # Data collection settings
    SEQ_LENGTH: int = _env_int("SEQ_LENGTH", 30)
    SECS_FOR_ACTION: int = _env_int("SECS_FOR_ACTION", 30)

    # Model settings
    MODEL_NAME: str = "model.h5"

    # Inference settings
    DEFAULT_THRESHOLD: float = _env_float("CONFIDENCE_THRESHOLD", 0.9)
    STABLE_COUNT: int = _env_int("STABLE_COUNT", 3)

    # Smart Thresholding: specific thresholds for critical actions
    SMART_THRESHOLDS: Dict[str, float] = field(default_factory=lambda: {
        "stop": 0.98,
        "emergency": 0.99
    })

    # Initial gestures (matching model.h5 output shape which is 2)
    ACTIONS: List[str] = field(default_factory=lambda: ["come", "away"])

    # CORS settings
    CORS_ALLOWED_ORIGINS: List[str] = field(default_factory=lambda:
        os.getenv(
            "CORS_ALLOWED_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8000,http://127.0.0.1:8000"
        ).split(",")
    )

    def get_threshold(self, action: str) -> float:
        return self.SMART_THRESHOLDS.get(action, self.DEFAULT_THRESHOLD)

config = GestureConfig()
