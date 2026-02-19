import numpy as np
import tensorflow as tf
from pathlib import Path
import json
import sys
import os

# Add project root to sys.path to find gesture_recognition package
project_root = Path(__file__).resolve().parent.parent / "Hand movements and gestures"
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from gesture_recognition.features import hand_landmarks_to_feature_vector
except ImportError:
    print("Warning: Could not import gesture_recognition.features. Define locally.")
    def hand_landmarks_to_feature_vector(landmarks):
        return np.zeros((99,), dtype=np.float32)

class GestureInference:
    def __init__(self, model_path="models/model.h5", actions_path="models/actions.json"):
        self.model_path = Path(model_path)
        self.actions_path = Path(actions_path)
        self.model = None
        self.actions = ["come", "away", "spin"] # Defaults
        self.seq_length = 30
        self.sequence = []
        self.threshold = 0.8

        self.load_model()

    def load_model(self):
        # Load actions if available
        if self.actions_path.exists():
            try:
                with open(self.actions_path, "r") as f:
                    self.actions = json.load(f)
                print(f"Loaded actions: {self.actions}")
            except Exception as e:
                print(f"Error loading actions: {e}")

        # Load model
        if self.model_path.exists():
            try:
                self.model = tf.keras.models.load_model(str(self.model_path), compile=False)
                print(f"Model loaded from {self.model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
                self.model = None
        else:
            print(f"Model not found at {self.model_path}")

    def predict(self, landmarks_result):
        """
        Update sequence with new landmarks and predict.
        Returns: (action_name, confidence) or (None, 0.0)
        """
        if not self.model:
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

        self.sequence.append(fv)
        self.sequence = self.sequence[-self.seq_length:]

        if len(self.sequence) == self.seq_length:
            input_data = np.expand_dims(self.sequence, axis=0)
            res = self.model.predict(input_data, verbose=0)[0]

            idx = np.argmax(res)
            confidence = res[idx]

            if confidence > self.threshold:
                if idx < len(self.actions):
                    return self.actions[idx], confidence

        return None, 0.0
