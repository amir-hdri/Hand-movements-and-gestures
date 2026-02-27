import numpy as np
import tensorflow as tf
from typing import List, Optional, Dict
from dataclasses import dataclass

from gesture_recognition.recognizer import GestureRecognizer, Prediction
from .config import config

class SmartGestureRecognizer(GestureRecognizer):
    def __init__(
        self,
        model,
        actions: List[str],
        seq_length: int = 30,
        threshold: float = 0.9,
        stable_count: int = 3,
        smart_thresholds: Dict[str, float] = None
    ):
        super().__init__(model, actions, seq_length=seq_length, threshold=threshold, stable_count=stable_count)
        self.smart_thresholds = smart_thresholds or {}

    def update(self, feature_vector: np.ndarray) -> Prediction:
        """Update with smart thresholding logic."""
        feature_vector = np.asarray(feature_vector, dtype=np.float32)
        self._seq.append(feature_vector)
        if len(self._seq) < self._seq_length:
            return Prediction(raw_action=None, confidence=0.0, stable_action=None)

        input_data = np.expand_dims(
            np.asarray(list(self._seq)[-self._seq_length :], dtype=np.float32), axis=0
        )

        try:
            y_pred = self._model.predict(input_data, verbose=0).squeeze()
        except TypeError:
            y_pred = self._model.predict(input_data).squeeze()

        # Handle scalar output (single class)
        if y_pred.ndim == 0:
            y_pred = np.array([y_pred])

        if y_pred.ndim != 1 or y_pred.size != len(self._actions):
             # Handle edge case where model output shape doesn't match actions
             # e.g. if actions list changed but model is old
             print(f"Model output mismatch: {y_pred.shape} vs {len(self._actions)}")
             return Prediction(raw_action=None, confidence=0.0, stable_action=None)

        i_pred = int(np.argmax(y_pred))
        conf = float(y_pred[i_pred])
        raw_action = self._actions[i_pred]

        # Smart Thresholding
        threshold = self.smart_thresholds.get(raw_action, self._threshold)

        if conf < threshold:
            return Prediction(raw_action=None, confidence=conf, stable_action=None)

        self._action_seq.append(raw_action)

        stable_action: Optional[str] = None
        if len(self._action_seq) >= self._stable_count:
            last = list(self._action_seq)[-self._stable_count :]
            if all(action == last[0] for action in last):
                stable_action = last[0]

        return Prediction(raw_action=raw_action, confidence=conf, stable_action=stable_action)

class InferenceEngine:
    def __init__(self):
        self.model = None
        self.recognizer = None
        self.actions = []
        self.model_path = config.MODELS_DIR / config.MODEL_NAME

    def load_model(self, actions: List[str]):
        if not self.model_path.exists():
            print(f"Model not found at {self.model_path}")
            return False

        print(f"Loading model from {self.model_path}")
        try:
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
            self.actions = actions
            self.recognizer = SmartGestureRecognizer(
                self.model,
                actions,
                seq_length=config.SEQ_LENGTH,
                threshold=config.DEFAULT_THRESHOLD,
                stable_count=config.STABLE_COUNT,
                smart_thresholds=config.SMART_THRESHOLDS
            )
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def predict(self, feature_vector):
        if not self.recognizer:
            return None
        return self.recognizer.update(feature_vector)

    def update_thresholds(self, smart_thresholds: Dict[str, float]):
        if self.recognizer:
            self.recognizer.smart_thresholds = smart_thresholds
