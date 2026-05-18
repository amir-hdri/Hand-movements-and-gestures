import unittest
from unittest.mock import Mock, patch
from pathlib import Path
import sys

# Safe mocking pattern for sys.modules that doesn't leak
class MockNumpy:
    ndarray = Mock()
    float32 = Mock()
    int32 = Mock()
    zeros = Mock()
    array = Mock()
    asarray = Mock()
    save = Mock()

class MockMediaPipe:
    class tasks:
        class python:
            class vision:
                pass
            class BaseOptions:
                pass
            class core:
                class base_options:
                    pass

class TestDataManager(unittest.TestCase):
    @patch.dict('sys.modules', {
        'numpy': MockNumpy(),
        'mediapipe': MockMediaPipe(),
        'mediapipe.tasks': MockMediaPipe.tasks,
        'mediapipe.tasks.python': MockMediaPipe.tasks.python,
        'mediapipe.tasks.python.vision': MockMediaPipe.tasks.python.vision,
    })
    def setUp(self):
        # Import inside the test to ensure it uses the patched sys.modules
        from gesture_recognition.gui.backend.data_manager import DataManager, LabelManager
        self.DataManager = DataManager
        self.LabelManager = LabelManager

    @patch.dict('sys.modules', {
        'numpy': MockNumpy(),
        'mediapipe': MockMediaPipe(),
        'mediapipe.tasks': MockMediaPipe.tasks,
        'mediapipe.tasks.python': MockMediaPipe.tasks.python,
        'mediapipe.tasks.python.vision': MockMediaPipe.tasks.python.vision,
    })
    @patch('gesture_recognition.gui.backend.data_manager.config')
    def test_start_recording(self, mock_config):
        mock_config.DATASET_DIR = Path('/tmp/mock_dataset')

        manager = self.DataManager()

        # Mock label manager methods
        manager.label_manager = Mock(spec=self.LabelManager)
        manager.label_manager.get_labels.return_value = ["idle"]

        # Test 1: Starting a recording with an existing label
        manager.start_recording("idle")
        self.assertTrue(manager.recording)
        self.assertEqual(manager.current_label, "idle")
        manager.label_manager.add_label.assert_not_called()
        self.assertEqual(manager.current_data, [])

        # Test 2: Starting a recording when already recording
        with self.assertRaises(RuntimeError) as cm:
            manager.start_recording("swipe")
        self.assertEqual(str(cm.exception), "Already recording")

        # Reset recording state manually for the next test
        manager.recording = False
        manager.label_manager.add_label.reset_mock()

        # Test 3: Starting a recording with a new label
        manager.label_manager.get_labels.return_value = ["idle"]
        manager.start_recording("swipe")
        self.assertTrue(manager.recording)
        self.assertEqual(manager.current_label, "swipe")
        manager.label_manager.add_label.assert_called_once_with("swipe")

if __name__ == "__main__":
    unittest.main()
