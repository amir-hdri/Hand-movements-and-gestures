"""Gesture recognition utilities for this project.

This package contains small, dependency-light helpers that are shared between
the dataset collector and the realtime demo scripts.
"""

from .features import (
    append_label,
    hand_landmarks_to_feature_vector,
    hand_landmarks_to_joint,
    joint_to_angles,
)
from .recognizer import GestureRecognizer

__all__ = [
    "GestureRecognizer",
    "append_label",
    "hand_landmarks_to_feature_vector",
    "hand_landmarks_to_joint",
    "joint_to_angles",
]

