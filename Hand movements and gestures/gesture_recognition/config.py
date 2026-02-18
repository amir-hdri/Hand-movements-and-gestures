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
    actions: List[str] = field(default_factory=lambda: ["come", "away", "spin"])
    seq_length: int = 30
    threshold: float = 0.9
    stable_count: int = 3

    # Output settings
    save_video: bool = True
    output_dir: Path = Path("artifacts/videos")

    # Dataset collection settings
    secs_for_action: int = 30
    dataset_output_dir: Path = Path("dataset")
