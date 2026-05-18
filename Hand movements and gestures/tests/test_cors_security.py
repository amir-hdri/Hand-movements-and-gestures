import unittest
import os
import sys
from unittest.mock import MagicMock

# Mock numpy and other dependencies that are not available in the test environment
# so we can test the config logic in isolation.
sys.modules['numpy'] = MagicMock()
sys.modules['cv2'] = MagicMock()
sys.modules['mediapipe'] = MagicMock()
sys.modules['mediapipe.tasks'] = MagicMock()
sys.modules['mediapipe.tasks.python'] = MagicMock()
sys.modules['mediapipe.tasks.python.vision'] = MagicMock()
sys.modules['mediapipe.framework.formats'] = MagicMock()
sys.modules['mediapipe.framework.formats.landmark_pb2'] = MagicMock()

class TestCORSSecurity(unittest.TestCase):
    def test_cors_allowed_origins_default(self):
        # Clear environment variable to test default
        if "CORS_ALLOWED_ORIGINS" in os.environ:
            del os.environ["CORS_ALLOWED_ORIGINS"]

        from gesture_recognition.gui.backend.config import GestureConfig
        config = GestureConfig()

        expected = [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:8000",
            "http://127.0.0.1:8000"
        ]
        self.assertEqual(config.CORS_ALLOWED_ORIGINS, expected)

    def test_cors_allowed_origins_env_override(self):
        os.environ["CORS_ALLOWED_ORIGINS"] = "https://example.com, https://myapp.com "

        from gesture_recognition.gui.backend.config import GestureConfig
        config = GestureConfig()

        expected = ["https://example.com", "https://myapp.com"]
        self.assertEqual(config.CORS_ALLOWED_ORIGINS, expected)

if __name__ == "__main__":
    unittest.main()
