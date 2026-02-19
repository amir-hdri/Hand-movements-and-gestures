from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Optional, Sequence, List

import numpy as np


@dataclass
class Prediction:
    """
    Result of a gesture recognition attempt.

    Attributes:
        raw_action: The action predicted by the model in the current frame, if confidence threshold is met.
        confidence: The confidence score of the prediction.
        stable_action: The action that has been stably predicted for `stable_count` consecutive frames.
    """
    raw_action: Optional[str]
    confidence: float
    stable_action: Optional[str]


class GestureRecognizer:
    """
    Realtime gesture recognizer using a sequence model (e.g., LSTM) and temporal smoothing.

    This class maintains a buffer of feature vectors and action predictions to provide stable
    gesture recognition results.
    """

    def __init__(
        self,
        model: Any,
        actions: Sequence[str],
        *,
        seq_length: int = 30,
        threshold: float = 0.85,
        stable_count: int = 3,
        critical_actions: Optional[List[str]] = None,
        critical_threshold: float = 0.95,
    ) -> None:
        """
        Initialize the GestureRecognizer.

        Args:
            model: A trained Keras/TensorFlow model (or compatible object with a .predict() method).
            actions: A sequence of action names corresponding to the model's output classes.
            seq_length: The number of frames (feature vectors) required for a single prediction.
            threshold: The minimum confidence score required to consider a prediction valid.
            stable_count: The number of consecutive consistent predictions required to report a stable action.
            critical_actions: List of actions that require higher confidence.
            critical_threshold: The threshold for critical actions.
        """
        if seq_length <= 0:
            raise ValueError("seq_length must be > 0")
        if stable_count <= 0:
            raise ValueError("stable_count must be > 0")

        self._model = model
        self._actions = list(actions)
        self._seq_length = int(seq_length)
        self._threshold = float(threshold)
        self._stable_count = int(stable_count)
        self._critical_actions = set(critical_actions) if critical_actions else set()
        self._critical_threshold = float(critical_threshold)

        # Buffer for feature vectors.
        self._seq: Deque[np.ndarray] = deque(maxlen=self._seq_length)

        # We need to buffer raw predictions to check stability
        self._prediction_buffer: Deque[Optional[str]] = deque(maxlen=self._stable_count)

    @property
    def seq_length(self) -> int:
        return self._seq_length

    def reset(self) -> None:
        """Clear the internal feature and action buffers."""
        self._seq.clear()
        self._prediction_buffer.clear()

    def update(self, feature_vector: np.ndarray) -> Prediction:
        """
        Update internal state with a new feature vector and return a prediction.

        Args:
            feature_vector: A 1D numpy array representing the features of the current frame.

        Returns:
            A Prediction object containing the raw action, confidence, and stable action (if any).
        """
        self._seq.append(np.asarray(feature_vector, dtype=np.float32))

        # Not enough history
        if len(self._seq) < self._seq_length:
            return Prediction(raw_action=None, confidence=0.0, stable_action=None)

        # Prepare input
        input_data = np.expand_dims(np.array(self._seq), axis=0)

        # Inference
        try:
            # TF/Keras predict returns (batch, classes)
            raw_pred = self._model.predict(input_data, verbose=0)
        except (TypeError, ValueError):
            # Fallback for some environments
            raw_pred = self._model.predict(input_data)

        y_pred = raw_pred[0] # First sample in batch
        i_pred = int(np.argmax(y_pred))
        conf = float(y_pred[i_pred])

        detected_action = None

        # Check thresholds
        if i_pred < len(self._actions):
            action_name = self._actions[i_pred]
            required_threshold = self._threshold

            if action_name in self._critical_actions:
                required_threshold = self._critical_threshold

            if conf >= required_threshold:
                detected_action = action_name

        # Update buffer for stability check
        self._prediction_buffer.append(detected_action)

        # Determine stable action
        stable_action = None
        if len(self._prediction_buffer) == self._stable_count:
            # Check if all recent predictions are the same and not None
            first = self._prediction_buffer[0]
            if first is not None and all(p == first for p in self._prediction_buffer):
                stable_action = first

        return Prediction(
            raw_action=detected_action,
            confidence=conf,
            stable_action=stable_action
        )
