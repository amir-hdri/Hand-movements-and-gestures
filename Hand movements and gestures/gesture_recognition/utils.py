import cv2
import logging
import sys
from pathlib import Path

def setup_logging(name: str = "GestureApp", level: int = logging.INFO) -> logging.Logger:
    """Setup standard logging to stdout."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def open_camera(source: str | int) -> cv2.VideoCapture:
    """
    Open a camera source.

    Args:
        source: Camera index (int) or video file path (str).

    Returns:
        cv2.VideoCapture object.

    Raises:
        RuntimeError: If the source cannot be opened.
    """
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera source: {source}")
    return cap
