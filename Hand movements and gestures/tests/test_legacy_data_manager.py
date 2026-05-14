import unittest
import sys
from unittest.mock import MagicMock, patch
import time

# Mock cv2, mediapipe, and numpy as we are running in an offline test environment
# and these libraries might not be available
sys.modules['cv2'] = MagicMock()
sys.modules['mediapipe'] = MagicMock()
sys.modules['numpy'] = MagicMock()

# Now we can import the DataCollector
try:
    from legacy_gui.data_manager import DataCollector
except ImportError:
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from legacy_gui.data_manager import DataCollector


class TestLegacyDataManager(unittest.TestCase):
    def setUp(self):
        # Prevent DataCollector from creating actual directories during __init__
        with patch('legacy_gui.data_manager.Path.mkdir'):
            self.collector = DataCollector(output_dir="dummy_dir")

    @patch('time.time')
    def test_start_recording(self, mock_time):
        mock_time.return_value = 12345.0

        # Initial state checks
        self.assertFalse(self.collector.is_recording)
        self.assertIsNone(self.collector.current_action)

        # Ensure there is some pre-existing data or state to check it gets cleared
        self.collector.recorded_data = [1, 2, 3]
        self.collector.frame_count = 10
        self.collector.start_time = 0

        # Test start_recording
        self.collector.start_recording("test_action")

        # Verification
        self.assertTrue(self.collector.is_recording)
        self.assertEqual(self.collector.current_action, "test_action")
        self.assertEqual(self.collector.recorded_data, [])
        self.assertEqual(self.collector.start_time, 12345.0)
        self.assertEqual(self.collector.frame_count, 0)

    @patch('time.time')
    def test_start_recording_while_recording(self, mock_time):
        mock_time.return_value = 12345.0

        # Set up an active recording session
        self.collector.is_recording = True
        self.collector.current_action = "existing_action"
        self.collector.recorded_data = [1, 2, 3]
        self.collector.frame_count = 3
        self.collector.start_time = 10000.0

        # Test start_recording again (should not overwrite state if already recording)
        self.collector.start_recording("new_action")

        # Verification: state should remain unchanged
        self.assertTrue(self.collector.is_recording)
        self.assertEqual(self.collector.current_action, "existing_action")
        self.assertEqual(self.collector.recorded_data, [1, 2, 3])
        self.assertEqual(self.collector.start_time, 10000.0)
        self.assertEqual(self.collector.frame_count, 3)

if __name__ == '__main__':
    unittest.main()
