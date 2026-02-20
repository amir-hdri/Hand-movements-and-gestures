import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
import logging
import asyncio
from pathlib import Path
from typing import Optional, List

# Add project root to sys.path
import sys
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from gesture_recognition.features import hand_landmarks_to_feature_vector
from gesture_recognition.recognizer import GestureRecognizer, Prediction
from gesture_recognition.gui.data_manager import DatasetManager
from gesture_recognition.gui.robot_interface import RobotController, PingPongController, MockRobotController

logger = logging.getLogger(__name__)

class InferenceEngine:
    def __init__(self, model_path: Optional[Path] = None, actions: List[str] = None):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.cap = None
        self.is_running = False
        self.camera_index = 0

        self.dataset_manager = DatasetManager()
        self.robot_controller: Optional[RobotController] = None

        self.recognizer: Optional[GestureRecognizer] = None
        self.model = None
        self.actions = actions if actions else []
        self.model_path = model_path

        self.current_prediction: Optional[Prediction] = None
        self.last_action_time = 0

        # Load model if provided
        if self.model_path and self.model_path.exists():
            self.load_model(self.model_path, self.actions)

    def load_model(self, model_path: Path, actions: List[str]):
        try:
            # Use tf.keras.models.load_model
            # Note: compile=False is often safer for inference only, especially with custom metrics/losses
            self.model = tf.keras.models.load_model(str(model_path), compile=False)
            self.actions = actions
            self.recognizer = GestureRecognizer(
                self.model,
                self.actions,
                seq_length=30,
                threshold=0.9,
                stable_count=3
            )
            logger.info(f"Model loaded from {model_path} with actions: {actions}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
            self.recognizer = None

    def start_camera(self, camera_index=0):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()

        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            logger.error(f"Could not open camera {camera_index}")
            return False

        self.is_running = True
        logger.info(f"Camera {camera_index} started")
        return True

    def stop_camera(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.cap = None
        logger.info("Camera stopped")

    def set_robot_controller(self, controller_type: str = "mock"):
        if self.robot_controller:
            self.robot_controller.disconnect()

        if controller_type == "pingpong":
            self.robot_controller = PingPongController()
        else:
            self.robot_controller = MockRobotController()

        try:
            self.robot_controller.connect()
        except Exception as e:
            logger.error(f"Failed to connect robot: {e}")
            self.robot_controller = None

    async def get_frame(self):
        if not self.is_running or self.cap is None:
            # Return a blank frame or None if not running
            # Wait a bit to simulate frame rate if stopped to prevent busy loop in consumer
            await asyncio.sleep(0.1)
            return None

        success, frame = self.cap.read()
        if not success:
            logger.warning("Failed to read frame")
            await asyncio.sleep(0.1)
            return None

        # 1. Recording Mode
        if self.dataset_manager.recording:
            frame = self.dataset_manager.process_frame(
                frame, self.hands, self.mp_drawing, self.mp_hands
            )
            # Encode for streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            return buffer.tobytes()

        # 2. Inference Mode
        img = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        prediction_text = "?"

        if result.multi_hand_landmarks:
            for res in result.multi_hand_landmarks:
                # Feature extraction
                if self.recognizer:
                    fv = hand_landmarks_to_feature_vector(res)
                    pred = self.recognizer.update(fv)
                    self.current_prediction = pred

                    if pred.stable_action:
                        prediction_text = pred.stable_action.upper()
                        # Robot Control
                        if self.robot_controller:
                            self.robot_controller.send_command(pred.stable_action)

                # Draw landmarks
                self.mp_drawing.draw_landmarks(img, res, self.mp_hands.HAND_CONNECTIONS)

        # Draw overlay
        cv2.putText(
            img,
            f"Action: {prediction_text}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        ret, buffer = cv2.imencode('.jpg', img)
        return buffer.tobytes()
