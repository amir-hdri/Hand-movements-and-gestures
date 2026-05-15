import unittest
from unittest.mock import patch, MagicMock

# We need to mock cv2, mediapipe, and numpy to avoid needing them installed
import sys

cv2_mock = MagicMock()
mp_mock = MagicMock()
np_mock = MagicMock()

sys.modules['cv2'] = cv2_mock
sys.modules['mediapipe'] = mp_mock
sys.modules['numpy'] = np_mock

import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from legacy_gui.data_manager import DataCollector
import time

class TestLegacyDataCollector(unittest.TestCase):
    def setUp(self):
        self.output_dir = "test_dataset"
        self.collector = DataCollector(output_dir=self.output_dir)

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
