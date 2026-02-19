import numpy as np
import tensorflow as tf
from pathlib import Path
import json
import sys
import os

from gesture_recognition.config import GestureConfig
from gesture_recognition.features import hand_landmarks_to_feature_vector
from gesture_recognition.recognizer import GestureRecognizer, Prediction

class GestureInference:
    def __init__(self, config: GestureConfig):
        self.config = config
        self.model_path = self.config.model_path
        self.actions_path = self.model_path.parent / "actions.json"

        self.model = None
        self.actions = self.config.actions # Start with defaults

        self.recognizer = None

        self.load_model()

    def load_model(self):
        # Load actions if available (overrides defaults)
        if self.actions_path.exists():
            try:
                with open(self.actions_path, "r") as f:
                    self.actions = json.load(f)
                print(f"Loaded actions from {self.actions_path}: {self.actions}")
            except Exception as e:
                print(f"Error loading actions: {e}")

        # Load model
        if self.model_path.exists():
            try:
                self.model = tf.keras.models.load_model(str(self.model_path), compile=False)
                print(f"Model loaded from {self.model_path}")

                # Initialize Recognizer
                self.recognizer = GestureRecognizer(
                    self.model,
                    self.actions,
                    seq_length=self.config.seq_length,
                    threshold=self.config.threshold,
                    stable_count=self.config.stable_count,
                    critical_actions=self.config.critical_actions,
                    critical_threshold=self.config.critical_threshold
                )

            except Exception as e:
                print(f"Error loading model: {e}")
                self.model = None
                self.recognizer = None
        else:
            print(f"Model not found at {self.model_path}. Inference disabled.")
            self.model = None
            self.recognizer = None

    def predict(self, landmarks_result):
        """
        Update sequence with new landmarks and predict.
        Returns: (action_name, confidence) or (None, 0.0)
        """
        if not self.recognizer:
            return None, 0.0

        if not landmarks_result.multi_hand_landmarks:
            return None, 0.0

        # Take the first hand
        res = landmarks_result.multi_hand_landmarks[0]
        try:
            fv = hand_landmarks_to_feature_vector(res)
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None, 0.0

        prediction: Prediction = self.recognizer.update(fv)

        # Return stable action if available, or maybe raw action if debugging?
        # The GUI expects (action, conf).
        # Using stable_action is safer for robot control.

        if prediction.stable_action:
            return prediction.stable_action, prediction.confidence

        # Optional: return raw action with lower confidence just for UI feedback?
        # For now, stick to stable action to avoid jitter.

        return None, 0.0
