import time
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

from .config import GestureConfig
from .features import hand_landmarks_to_feature_vector
from .recognizer import GestureRecognizer, Prediction
from .utils import setup_logging

logger = setup_logging(__name__)

class GesturePipeline:
    """
    Pipeline for Hand Gesture Recognition.

    Handles:
    1. Model loading
    2. MediaPipe Hands initialization
    3. Frame processing (Flip -> Detect -> Extract Features -> Recognize -> Draw)
    """

    def __init__(self, config: GestureConfig):
        self.config = config

        # Load Model
        if not self.config.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.config.model_path}")

        logger.info(f"Loading model from {self.config.model_path}")
        self.model = tf.keras.models.load_model(str(self.config.model_path), compile=False)

        # Initialize Recognizer
        self.recognizer = GestureRecognizer(
            self.model,
            self.config.actions,
            seq_length=self.config.seq_length,
            threshold=self.config.threshold,
            stable_count=self.config.stable_count,
            critical_actions=self.config.critical_actions,
            critical_threshold=self.config.critical_threshold,
        )

        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.last_overlay_text = "?"
        self.last_overlay_time = 0.0

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[Prediction]]:
        """
        Process a single frame: flip, detect hands, recognize gesture, draw landmarks.

        Args:
            frame: Raw frame from camera (BGR).

        Returns:
            processed_frame: The frame with drawings (if any).
            prediction: The prediction result (if hand detected).
        """
        # Flip frame horizontally for selfie-view display
        img = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        prediction: Optional[Prediction] = None

        if result.multi_hand_landmarks:
            for res in result.multi_hand_landmarks:
                # Feature extraction
                fv = hand_landmarks_to_feature_vector(res)

                # Recognition
                prediction = self.recognizer.update(fv)

                # Draw landmarks
                self.mp_drawing.draw_landmarks(img, res, self.mp_hands.HAND_CONNECTIONS)

                # Update overlay text logic
                if prediction.stable_action:
                    self.last_overlay_text = prediction.stable_action.upper()
                    self.last_overlay_time = time.time()

                # Draw text near hand
                anchor = res.landmark[0] # Wrist
                cv2.putText(
                    img,
                    self.last_overlay_text,
                    org=(int(anchor.x * img.shape[1]), int(anchor.y * img.shape[0] + 20)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(255, 255, 255),
                    thickness=2,
                )

        # Fade overlay text if no stable action recently
        if time.time() - self.last_overlay_time > 1.5:
             self.last_overlay_text = "?"

        return img, prediction

    def close(self):
        """Release resources."""
        self.hands.close()
