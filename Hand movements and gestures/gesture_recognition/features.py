from __future__ import annotations

from typing import Protocol, Sequence, Any

import numpy as np


class Landmark(Protocol):
    x: float
    y: float
    z: float


class HandLandmarks(Protocol):
    landmark: Sequence[Landmark]


_PARENT_JOINTS = np.array(
    [0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19],
    dtype=np.int32,
)
_CHILD_JOINTS = np.array(
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    dtype=np.int32,
)

_ANGLE_V1 = np.array([0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], dtype=np.int32)
_ANGLE_V2 = np.array([1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], dtype=np.int32)


def hand_landmarks_to_joint(hand_landmarks: Any) -> np.ndarray:
    """Convert a MediaPipe-like HandLandmarks object into a (21, 4) float32 array.

    The 4th column (visibility) is optional in MediaPipe Hands; if missing, it's set to 0.
    """

    if hasattr(hand_landmarks, "landmark"):
        landmarks = hand_landmarks.landmark
    else:
        # Assume it's a list/sequence of landmarks (New API)
        landmarks = hand_landmarks

    if len(landmarks) != 21:
        raise ValueError(f"Expected 21 landmarks, got {len(landmarks)}")

    joint = np.zeros((21, 4), dtype=np.float32)
    for index, lm in enumerate(landmarks):
        joint[index, 0] = float(lm.x)
        joint[index, 1] = float(lm.y)
        joint[index, 2] = float(lm.z)
        joint[index, 3] = float(getattr(lm, "visibility", 0.0))

    return joint


def joint_to_angles(joint: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    """Compute 15 joint angles from a (21, >=3) joint array.

    Returns a (15,) float32 array in degrees.
    """

    if joint.shape[0] != 21 or joint.shape[1] < 3:
        raise ValueError(f"Expected joint shape (21, >=3), got {joint.shape}")

    v1 = joint[_PARENT_JOINTS, :3]
    v2 = joint[_CHILD_JOINTS, :3]
    v = v2 - v1

    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.where(norms < eps, eps, norms)
    v = v / norms

    dots = np.einsum("nt,nt->n", v[_ANGLE_V1, :], v[_ANGLE_V2, :])
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.degrees(np.arccos(dots)).astype(np.float32)
    return angles


def hand_landmarks_to_feature_vector(hand_landmarks: Any) -> np.ndarray:
    """Build a (99,) float32 feature vector for model inference.

    Layout matches the original training setup:
    - 21 landmarks * 4 values (x, y, z, visibility) => 84 values
    - 15 joint angles (degrees) => 15 values
    Total: 99 values.
    """

    joint = hand_landmarks_to_joint(hand_landmarks)
    angles = joint_to_angles(joint)
    return np.concatenate([joint.flatten(), angles], axis=0).astype(np.float32)


def append_label(feature_vector: np.ndarray, label_index: int) -> np.ndarray:
    """Append a numeric label to a feature vector as the last element."""

    feature_vector = np.asarray(feature_vector, dtype=np.float32)
    label = np.asarray([float(label_index)], dtype=np.float32)
    return np.concatenate([feature_vector, label], axis=0)
