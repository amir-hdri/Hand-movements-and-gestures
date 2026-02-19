import unittest
from pathlib import Path
from gesture_recognition.config import GestureConfig

class TestConfig(unittest.TestCase):
    def test_defaults(self):
        config = GestureConfig()
        self.assertEqual(config.camera_id, 0)
        self.assertEqual(config.seq_length, 30)
        self.assertTrue(config.save_video)
        self.assertEqual(config.actions, ["come", "away", "spin", "thumbs_up", "peace", "fist", "stop"])

if __name__ == "__main__":
    unittest.main()
