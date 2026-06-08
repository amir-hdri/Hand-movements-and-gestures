import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestLegacyDataCollector(unittest.TestCase):
    def setUp(self):
        # Create mocks for the required modules
        cv2_mock = MagicMock()
        np_mock = MagicMock()
        mp_mock = MagicMock()
        mp_mock.solutions = MagicMock()
        mp_mock.solutions.hands = MagicMock()
        mp_mock.solutions.hands.Hands = MagicMock()
        
        # Import under the patched sys.modules
        with patch.dict('sys.modules', {
            'cv2': cv2_mock,
            'mediapipe': mp_mock,
            'numpy': np_mock
        }):
            from legacy_gui.data_manager import DataCollector
            self.DataCollector = DataCollector
        
        self.output_dir = "test_dataset"
        self.collector = self.DataCollector(output_dir=self.output_dir)

    def test_start_recording_not_recording(self):
        """Test starting recording when not currently recording."""
        self.collector.is_recording = False
        self.collector.recorded_data = [1, 2, 3] # simulate dirty state
        self.collector.frame_count = 10
        self.collector.start_time = 0

        self.collector.start_recording("test_action")

        self.assertTrue(self.collector.is_recording)
        self.assertEqual(self.collector.current_action, "test_action")
        self.assertEqual(self.collector.recorded_data, [])
        self.assertEqual(self.collector.frame_count, 0)
        self.assertNotEqual(self.collector.start_time, 0)

    def test_start_recording_already_recording(self):
        """Test starting recording when already recording (should not reset state)."""
        self.collector.is_recording = True
        self.collector.current_action = "existing_action"
        self.collector.recorded_data = [1, 2, 3]
        self.collector.frame_count = 10
        initial_time = 100.0
        self.collector.start_time = initial_time

        self.collector.start_recording("new_action")

        # State should remain unchanged
        self.assertTrue(self.collector.is_recording)
        self.assertEqual(self.collector.current_action, "existing_action")
        self.assertEqual(self.collector.recorded_data, [1, 2, 3])
        self.assertEqual(self.collector.frame_count, 10)
        self.assertEqual(self.collector.start_time, initial_time)

if __name__ == "__main__":
    unittest.main()
