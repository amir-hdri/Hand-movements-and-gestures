import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import time
import numpy as np
import shutil
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestLegacyDataCollector(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        
        # Patch mediapipe so we don't load actual models
        self.mp_patcher = patch('legacy_gui.data_manager.mp')
        self.mock_mp = self.mp_patcher.start()
        
        from legacy_gui.data_manager import DataCollector
        self.DataCollector = DataCollector
        self.collector = self.DataCollector(output_dir=self.test_dir)

    def tearDown(self):
        self.mp_patcher.stop()
        shutil.rmtree(self.test_dir)

    def test_init(self):
        self.assertEqual(self.collector.output_dir, Path(self.test_dir))
        self.assertTrue(self.collector.output_dir.exists())
        self.assertFalse(self.collector.is_recording)
        self.assertIsNone(self.collector.current_action)
        self.assertEqual(self.collector.recorded_data, [])
        self.assertEqual(self.collector.frame_count, 0)

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

    @patch('legacy_gui.data_manager.DataCollector.save_data')
    def test_stop_recording(self, mock_save_data):
        # Stop when not recording
        self.assertEqual(self.collector.stop_recording(), 0)
        mock_save_data.assert_not_called()

        # Start and stop with no data
        self.collector.start_recording("test_action")
        self.assertEqual(self.collector.stop_recording(), 0)
        self.assertFalse(self.collector.is_recording)
        mock_save_data.assert_not_called()

        # Start and stop with data
        self.collector.start_recording("test_action")
        self.collector.recorded_data = [np.zeros(99)]
        self.assertEqual(self.collector.stop_recording(), 1)
        self.assertFalse(self.collector.is_recording)
        mock_save_data.assert_called_once()

    @patch('legacy_gui.data_manager.time.time')
    def test_save_data_no_sequences(self, mock_time):
        mock_time.return_value = 12345
        self.collector.current_action = "test_action"

        # Add 10 frames (less than seq_length 30)
        self.collector.recorded_data = [np.zeros((99,), dtype=np.float32) for _ in range(10)]
        self.collector.save_data()

        raw_file = Path(self.test_dir) / "raw_test_action_12345.npy"
        seq_file = Path(self.test_dir) / "seq_test_action_12345.npy"

        self.assertTrue(raw_file.exists())
        self.assertFalse(seq_file.exists())

        saved_raw = np.load(raw_file)
        self.assertEqual(saved_raw.shape, (10, 99))

    @patch('legacy_gui.data_manager.time.time')
    def test_save_data_with_sequences(self, mock_time):
        mock_time.return_value = 12345
        self.collector.current_action = "test_action"

        # Add 40 frames (more than seq_length 30)
        self.collector.recorded_data = [np.zeros((99,), dtype=np.float32) for _ in range(40)]
        self.collector.save_data()

        raw_file = Path(self.test_dir) / "raw_test_action_12345.npy"
        seq_file = Path(self.test_dir) / "seq_test_action_12345.npy"

        self.assertTrue(raw_file.exists())
        self.assertTrue(seq_file.exists())

        saved_raw = np.load(raw_file)
        self.assertEqual(saved_raw.shape, (40, 99))

        saved_seq = np.load(seq_file)
        self.assertEqual(saved_seq.shape, (11, 30, 99)) # 40 - 30 + 1 = 11 sliding windows

    @patch('legacy_gui.data_manager.cv2')
    @patch('legacy_gui.data_manager.hand_landmarks_to_feature_vector')
    def test_process_frame_no_hands(self, mock_feature_vector, mock_cv2):
        # Setup mock frame
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Setup mock cv2
        mock_cv2.flip.return_value = mock_frame
        mock_cv2.cvtColor.return_value = mock_frame

        # Setup mock hands process to return no landmarks
        mock_results = MagicMock()
        mock_results.multi_hand_landmarks = None
        self.collector.hands.process.return_value = mock_results

        img, results = self.collector.process_frame(mock_frame)

        self.assertEqual(results, mock_results)
        self.assertEqual(self.collector.frame_count, 0)
        self.assertEqual(self.collector.recorded_data, [])
        mock_feature_vector.assert_not_called()

    @patch('legacy_gui.data_manager.cv2')
    @patch('legacy_gui.data_manager.hand_landmarks_to_feature_vector')
    def test_process_frame_with_hands_not_recording(self, mock_feature_vector, mock_cv2):
        # Setup mock frame
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Setup mock cv2
        mock_cv2.flip.return_value = mock_frame
        mock_cv2.cvtColor.return_value = mock_frame

        # Setup mock hands process to return landmarks
        mock_results = MagicMock()
        mock_results.multi_hand_landmarks = [MagicMock()]
        self.collector.hands.process.return_value = mock_results

        img, results = self.collector.process_frame(mock_frame)

        self.assertEqual(results, mock_results)
        self.assertEqual(self.collector.frame_count, 0)
        self.assertEqual(self.collector.recorded_data, [])
        mock_feature_vector.assert_not_called()
        self.collector.mp_drawing.draw_landmarks.assert_called_once()

    @patch('legacy_gui.data_manager.cv2')
    @patch('legacy_gui.data_manager.hand_landmarks_to_feature_vector')
    def test_process_frame_with_hands_recording(self, mock_feature_vector, mock_cv2):
        self.collector.start_recording("test_action")

        # Setup mock frame
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Setup mock cv2
        mock_cv2.flip.return_value = mock_frame
        mock_cv2.cvtColor.return_value = mock_frame

        # Setup mock hands process to return landmarks
        mock_results = MagicMock()
        mock_results.multi_hand_landmarks = [MagicMock()]
        self.collector.hands.process.return_value = mock_results

        mock_feature_vector.return_value = np.ones((99,), dtype=np.float32)

        img, results = self.collector.process_frame(mock_frame)

        self.assertEqual(results, mock_results)
        self.assertEqual(self.collector.frame_count, 1)
        self.assertEqual(len(self.collector.recorded_data), 1)
        self.assertTrue(np.array_equal(self.collector.recorded_data[0], np.ones((99,), dtype=np.float32)))
        mock_feature_vector.assert_called_once()
        self.collector.mp_drawing.draw_landmarks.assert_called_once()
        mock_cv2.putText.assert_called_once()

if __name__ == "__main__":
    unittest.main()
