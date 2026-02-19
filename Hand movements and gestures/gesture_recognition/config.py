from dataclasses import dataclass, field
from pathlib import Path
from typing import List

@dataclass
class GestureConfig:
    """Configuration for Gesture Recognition Application."""

    # Camera settings
    camera_id: int = 0

    # Model settings
    model_path: Path = Path("models/model2_1.0.h5")
    # Expanded gesture set for better control and reliability
    actions: List[str] = field(default_factory=lambda: [
        "come",
        "away",
        "spin",
        "thumbs_up",
        "peace",
        "fist",
        "stop"
    ])
    seq_length: int = 30
    threshold: float = 0.85  # Slightly lower default threshold for broader acceptance, handled by smart logic
    stable_count: int = 3 # Number of frames to confirm a gesture

    # Output settings
    save_video: bool = True
    output_dir: Path = Path("artifacts/videos")

    # Dataset collection settings
    secs_for_action: int = 30
    dataset_output_dir: Path = Path("dataset")

    # Robot Control Settings
    # Actions that require higher confidence or stability
    critical_actions: List[str] = field(default_factory=lambda: ["stop", "spin"])
    critical_threshold: float = 0.95
