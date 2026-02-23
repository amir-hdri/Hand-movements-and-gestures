from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict

@dataclass
class GestureConfig:
    # Project paths
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent.parent
    DATASET_DIR: Path = PROJECT_ROOT / "dataset"
    MODELS_DIR: Path = PROJECT_ROOT / "models"

    # Data collection settings
    SEQ_LENGTH: int = 30
    SECS_FOR_ACTION: int = 30

    # Model settings
    MODEL_NAME: str = "model.h5"

    # Inference settings
    DEFAULT_THRESHOLD: float = 0.9
    STABLE_COUNT: int = 3

    # Smart Thresholding: specific thresholds for critical actions
    SMART_THRESHOLDS: Dict[str, float] = field(default_factory=lambda: {
        "stop": 0.98,
        "emergency": 0.99
    })

    # Initial gestures (matching model.h5 output shape which is 2)
    ACTIONS: List[str] = field(default_factory=lambda: ["come", "away"])

    def get_threshold(self, action: str) -> float:
        return self.SMART_THRESHOLDS.get(action, self.DEFAULT_THRESHOLD)

config = GestureConfig()
