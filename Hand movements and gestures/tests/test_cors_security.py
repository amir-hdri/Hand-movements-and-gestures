import unittest
import os
import sys
from pathlib import Path
from importlib import reload
from unittest.mock import MagicMock

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Mock heavy dependencies
mock_modules = [
    'numpy',
    'cv2',
    'mediapipe',
    'mediapipe.tasks',
    'mediapipe.tasks.python',
    'mediapipe.tasks.python.vision',
    'mediapipe.framework.formats',
    'mediapipe.framework.formats.landmark_pb2',
    'fastapi',
    'fastapi.responses',
    'fastapi.middleware.cors',
    'fastapi.staticfiles',
    'pydantic',
    'tensorflow',
    'tensorflow.keras',
    'tensorflow.keras.models',
    'tensorflow.keras.layers',
    'tensorflow.keras.callbacks',
    'tensorflow.keras.utils',
    'sklearn',
    'sklearn.model_selection',
    'sklearn.metrics'
]

for mod_name in mock_modules:
    sys.modules[mod_name] = MagicMock()

# Setup some specific mocks for FastAPI/Pydantic that are used in app.py
import fastapi
import fastapi.middleware.cors
import pydantic

class MockBaseModel:
    pass

pydantic.BaseModel = MockBaseModel

class TestCORSSecurity(unittest.TestCase):
    def setUp(self):
        # Clear environment variable before each test
        if "CORS_ALLOWED_ORIGINS" in os.environ:
            del os.environ["CORS_ALLOWED_ORIGINS"]

        # Reload the config module to ensure clean state
        import gesture_recognition.gui.backend.config as config
        reload(config)
        self.config_module = config

    def test_default_origins(self):
        """Test that default CORS origins are correctly set."""
        expected_defaults = [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:8000",
            "http://127.0.0.1:8000"
        ]
        self.assertEqual(self.config_module.config.CORS_ALLOWED_ORIGINS, expected_defaults)

    def test_env_variable_override(self):
        """Test that environment variable correctly overrides default CORS origins."""
        custom_origins = "https://myapp.com,https://api.myapp.com"
        os.environ["CORS_ALLOWED_ORIGINS"] = custom_origins

        # Reload to pick up env var
        reload(self.config_module)

        expected_origins = ["https://myapp.com", "https://api.myapp.com"]
        self.assertEqual(self.config_module.config.CORS_ALLOWED_ORIGINS, expected_origins)

if __name__ == "__main__":
    unittest.main()
