import unittest
import sys
import os

# Add the project directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestCORS(unittest.TestCase):
    def test_cors_config_file(self):
        # Read the file and check for the correct origins
        app_path = os.path.join(os.path.dirname(__file__), '..', 'gesture_recognition', 'gui', 'backend', 'app.py')
        with open(app_path, 'r') as f:
            content = f.read()

        expected_origins = [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
        ]

        # Check that we are not using wildcard
        self.assertNotIn('allow_origins=["*"]', content)

        # Check that all expected origins are present
        for origin in expected_origins:
            self.assertIn(origin, content)

if __name__ == "__main__":
    unittest.main()
