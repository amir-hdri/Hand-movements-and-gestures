from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Optional, Sequence

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
        threshold: float = 0.9,
        stable_count: int = 3,
    ) -> None:
        """
        Initialize the GestureRecognizer.

        Args:
            model: A trained Keras/TensorFlow model (or compatible object with a .predict() method).
            actions: A sequence of action names corresponding to the model's output classes.
            seq_length: The number of frames (feature vectors) required for a single prediction.
            threshold: The minimum confidence score required to consider a prediction valid.
            stable_count: The number of consecutive consistent predictions required to report a stable action.
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

        # Buffer for feature vectors.
        self._seq: Deque[np.ndarray] = deque(maxlen=self._seq_length)

        # Buffer for recent raw actions to determine stability.
        self._action_seq: Deque[Optional[str]] = deque(maxlen=self._stable_count)

    @property
    def seq_length(self) -> int:
        return self._seq_length

    def reset(self) -> None:
        """Clear the internal feature and action buffers."""
        self._seq.clear()
        self._action_seq.clear()

    def update(self, feature_vector: np.ndarray) -> Prediction:
        """
        Update internal state with a new feature vector and return a prediction.

        Args:
            feature_vector: A 1D numpy array representing the features of the current frame.

        Returns:
            A Prediction object containing the raw action, confidence, and stable action (if any).
        """
        feature_vector = np.asarray(feature_vector, dtype=np.float32)
        self._seq.append(feature_vector)

        # If we don't have enough history, we can't predict yet.
        if len(self._seq) < self._seq_length:
            return Prediction(raw_action=None, confidence=0.0, stable_action=None)

        # Prepare input for the model: (1, seq_length, num_features)
        # Note: np.array(deque) copies data. This is inevitable unless we pre-allocate a buffer
        # and roll it, but deque is cleaner for now.
        input_data = np.expand_dims(
            np.array(self._seq, dtype=np.float32), axis=0
        )

        # Handle different Keras versions/APIs
        try:
            y_pred = self._model.predict(input_data, verbose=0).squeeze()
        except TypeError:
            y_pred = self._model.predict(input_data).squeeze()

        if y_pred.ndim == 0:
             y_pred = np.atleast_1d(y_pred)

        i_pred = int(np.argmax(y_pred))
        conf = float(y_pred[i_pred])

        if conf < self._threshold:
            self._action_seq.append(None)
            return Prediction(raw_action=None, confidence=conf, stable_action=None)

        raw_action = self._actions[i_pred]
        self._action_seq.append(raw_action)

        stable_action: Optional[str] = None
        if len(self._action_seq) == self._stable_count:
            if all(action == self._action_seq[0] and action is not None for action in self._action_seq):
                stable_action = self._action_seq[0]

        return Prediction(raw_action=raw_action, confidence=conf, stable_action=stable_action)
