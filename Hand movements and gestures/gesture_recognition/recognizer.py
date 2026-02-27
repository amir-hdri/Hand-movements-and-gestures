from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Optional, Sequence

import numpy as np


@dataclass
class Prediction:
    raw_action: Optional[str]
    confidence: float
    stable_action: Optional[str]


class GestureRecognizer:
    """Realtime gesture recognizer with simple temporal smoothing."""

    def __init__(
        self,
        model: Any,
        actions: Sequence[str],
        *,
        seq_length: int = 30,
        threshold: float = 0.9,
        stable_count: int = 3,
    ) -> None:
        if seq_length <= 0:
            raise ValueError("seq_length must be > 0")
        if stable_count <= 0:
            raise ValueError("stable_count must be > 0")

        try:
            import tensorflow as tf
            # For massive performance gains in TF 2.x on single samples, we can
            # trace the Keras model execution graph.
            self._model = tf.function(model, reduce_retracing=True)
        except (ImportError, TypeError, ValueError, AttributeError):
            # Fallback if model is not compatible with tf.function or tf is missing
            self._model = model

        self._actions = list(actions)
        self._seq_length = int(seq_length)
        self._threshold = float(threshold)
        self._stable_count = int(stable_count)

        self._seq: Deque[np.ndarray] = deque(maxlen=self._seq_length * 3)
        self._action_seq: Deque[str] = deque(maxlen=self._stable_count * 3)

    @property
    def seq_length(self) -> int:
        return self._seq_length

    def update(self, feature_vector: np.ndarray) -> Prediction:
        """Update internal state with a new feature vector and return a prediction."""

        feature_vector = np.asarray(feature_vector, dtype=np.float32)
        self._seq.append(feature_vector)
        if len(self._seq) < self._seq_length:
            return Prediction(raw_action=None, confidence=0.0, stable_action=None)

        input_data = np.expand_dims(
            np.asarray(list(self._seq)[-self._seq_length :], dtype=np.float32), axis=0
        )

        # For single samples, invoking __call__ directly (or the tf.function wrapper)
        # avoids the massive batching overhead of .predict()
        y_pred = self._model(input_data, training=False)

        # Convert tensor to numpy and squeeze batch dimension
        if hasattr(y_pred, "numpy"):
            y_pred = y_pred.numpy()
        y_pred = np.asarray(y_pred).squeeze()

        if y_pred.ndim != 1 or y_pred.size != len(self._actions):
            raise ValueError(
                f"Unexpected model output shape {y_pred.shape}; expected ({len(self._actions)},)"
            )

        i_pred = int(np.argmax(y_pred))
        conf = float(y_pred[i_pred])
        if conf < self._threshold:
            return Prediction(raw_action=None, confidence=conf, stable_action=None)

        raw_action = self._actions[i_pred]
        self._action_seq.append(raw_action)

        stable_action: Optional[str] = None
        if len(self._action_seq) >= self._stable_count:
            last = list(self._action_seq)[-self._stable_count :]
            if all(action == last[0] for action in last):
                stable_action = last[0]

        return Prediction(raw_action=raw_action, confidence=conf, stable_action=stable_action)

