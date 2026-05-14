import unittest
from unittest.mock import patch, MagicMock

import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import mediapipe
# Mock the specific submodule to prevent ModuleNotFoundError
sys.modules['mediapipe.framework'] = MagicMock()
sys.modules['mediapipe.framework.formats'] = MagicMock()
sys.modules['mediapipe.framework.formats.landmark_pb2'] = MagicMock()
mediapipe.solutions = MagicMock()
mediapipe.solutions.drawing_utils = MagicMock()
mediapipe.solutions.hands = MagicMock()
sys.modules['mediapipe.solutions'] = mediapipe.solutions

mock_cap = MagicMock()
mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))

mock_landmarker = MagicMock()
mock_result = MagicMock()
mock_result.hand_landmarks = []
mock_landmarker.detect.return_value = mock_result

# We need to mock CameraThread completely before importing app
with patch("cv2.VideoCapture", return_value=mock_cap), \
     patch("mediapipe.tasks.python.vision.HandLandmarker.create_from_options", return_value=mock_landmarker):

    # We patch threading.Thread so that the base class doesn't do anything
    # Alternatively we can patch the CameraThread directly inside app if we had access to it
    # We will inject a dummy into sys.modules to patch

    # But first, we can just replace the CameraThread initialization
    pass

import importlib
with patch("cv2.VideoCapture", return_value=mock_cap), \
     patch("mediapipe.tasks.python.vision.HandLandmarker.create_from_options", return_value=mock_landmarker):
    import gesture_recognition.gui.backend.app as app_module

    # Override the background thread class to prevent it from running
    class DummyCameraThread:
        def __init__(self):
            self.daemon = True
        def start(self):
            pass
        def stop(self):
            pass

    # Swap out the class
    app_module.CameraThread = DummyCameraThread

    # Reload the module to run initialization with the dummy thread
    # Wait, reload will still run the module level code
    pass

with patch("cv2.VideoCapture", return_value=mock_cap), \
     patch("mediapipe.tasks.python.vision.HandLandmarker.create_from_options", return_value=mock_landmarker), \
     patch("threading.Thread.start", MagicMock()):
    from fastapi.testclient import TestClient
    from gesture_recognition.gui.backend.app import app, state

class TestAppStatus(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_get_status_idle(self):
        # Set to known default values
        state.mode = "idle"
        state.training_status = "idle"
        state.last_prediction = {"action": None, "confidence": 0.0}

        response = self.client.get("/api/status")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "mode": "idle",
            "training_status": "idle",
            "last_prediction": {"action": None, "confidence": 0.0}
        })

    def test_get_status_active(self):
        # Set to active values
        state.mode = "predicting"
        state.training_status = "completed"
        state.last_prediction = {"action": "swipe", "confidence": 0.98}

        response = self.client.get("/api/status")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "mode": "predicting",
            "training_status": "completed",
            "last_prediction": {"action": "swipe", "confidence": 0.98}
        })

if __name__ == "__main__":
    unittest.main()
