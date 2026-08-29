import unittest
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
import sys

# Create a master mock object generator
class MyMagicMock(MagicMock):
    def __iter__(self):
        return iter([])

# Create a mock for mediapipe package which behaves properly for __path__ etc.
mp_mock = MagicMock()
mp_python_mock = MagicMock()
mp_python_solutions_mock = MagicMock()
mp_python_solutions_mock.drawing_utils = MagicMock()
mp_python_solutions_mock.hands = MagicMock()

class TestAppStatus(unittest.TestCase):
    @patch.dict('sys.modules', {
        'numpy': MyMagicMock(),
        'mediapipe': mp_mock,
        'mediapipe.tasks': MyMagicMock(),
        'mediapipe.tasks.python': MyMagicMock(),
        'mediapipe.tasks.python.vision': MyMagicMock(),
        'mediapipe.framework': MyMagicMock(),
        'mediapipe.framework.formats': MyMagicMock(),
        'mediapipe.framework.formats.landmark_pb2': MyMagicMock(),
        'mediapipe.python': mp_python_mock,
        'mediapipe.python.solutions': mp_python_solutions_mock,
        'cv2': MyMagicMock(),
        'tensorflow': MyMagicMock(),
        'tensorflow.keras': MyMagicMock(),
        'tensorflow.keras.models': MyMagicMock(),
        'tensorflow.keras.layers': MyMagicMock(),
        'tensorflow.keras.callbacks': MyMagicMock(),
        'tensorflow.keras.utils': MyMagicMock(),
        'tensorflow.lite': MyMagicMock(),
        'sklearn': MyMagicMock(),
        'sklearn.model_selection': MyMagicMock(),
        'tf_keras': MyMagicMock(),
        'matplotlib': MyMagicMock(),
        'matplotlib.pyplot': MyMagicMock(),
    })
    def setUp(self):
        # We need to mock background thread creation in app.py
        with patch('threading.Thread.start'):
            # Also mock loading models etc
            from gesture_recognition.gui.backend.app import app, state
            self.app = app
            self.state = state
            self.client = TestClient(app)

    def test_get_status_idle(self):
        # Set up state
        self.state.mode = "idle"
        self.state.training_status = "idle"
        self.state.last_prediction = {"action": None, "confidence": 0.0}

        # Make request
        response = self.client.get("/api/status")

        # Verify response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["mode"], "idle")
        self.assertEqual(data["training_status"], "idle")
        self.assertEqual(data["last_prediction"]["action"], None)
        self.assertEqual(data["last_prediction"]["confidence"], 0.0)

    def test_get_status_recording(self):
        # Set up state
        self.state.mode = "recording"
        self.state.training_status = "idle"
        self.state.last_prediction = {"action": "swipe", "confidence": 0.85}

        # Make request
        response = self.client.get("/api/status")

        # Verify response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["mode"], "recording")
        self.assertEqual(data["training_status"], "idle")
        self.assertEqual(data["last_prediction"]["action"], "swipe")
        self.assertEqual(data["last_prediction"]["confidence"], 0.85)

    def test_get_status_training(self):
        # Set up state
        self.state.mode = "training"
        self.state.training_status = "training"
        self.state.last_prediction = {"action": None, "confidence": 0.0}

        # Make request
        response = self.client.get("/api/status")

        # Verify response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["mode"], "training")
        self.assertEqual(data["training_status"], "training")

if __name__ == "__main__":
    unittest.main()
