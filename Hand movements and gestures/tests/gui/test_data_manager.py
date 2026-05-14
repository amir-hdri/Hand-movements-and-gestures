import unittest
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add the project root to sys.path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Mock numpy, mediapipe etc to run offline
sys.modules['numpy'] = MagicMock()
sys.modules['mediapipe'] = MagicMock()
sys.modules['mediapipe.tasks'] = MagicMock()
sys.modules['mediapipe.tasks.python'] = MagicMock()
sys.modules['mediapipe.tasks.python.vision'] = MagicMock()

from gesture_recognition.gui.backend.data_manager import LabelManager

class TestLabelManager(unittest.TestCase):
    def setUp(self):
        self.dataset_dir = MagicMock(spec=Path)
        self.labels_file = MagicMock(spec=Path)
        self.dataset_dir.__truediv__.return_value = self.labels_file
        self.labels_file.exists.return_value = False

        # We need to mock config.ACTIONS since we fall back to it
        with patch('gesture_recognition.gui.backend.data_manager.config') as mock_config:
            mock_config.ACTIONS = ["action1", "action2"]
            self.label_manager = LabelManager(self.dataset_dir)
            # Ensure the labels are set correctly for our test
            self.label_manager.labels = ["action1", "action2"]

    def test_get_label_index_success(self):
        """Test getting the index of an existing label."""
        self.assertEqual(self.label_manager.get_label_index("action1"), 0)
        self.assertEqual(self.label_manager.get_label_index("action2"), 1)

    def test_get_label_index_error_path(self):
        """Test that ValueError is raised when getting index of a non-existent label."""
        with self.assertRaises(ValueError) as context:
            self.label_manager.get_label_index("nonexistent_action")

        self.assertEqual(str(context.exception), "Label nonexistent_action not found")

if __name__ == "__main__":
    unittest.main()
