import unittest
from unittest.mock import Mock, MagicMock
import numpy as np

from gesture_recognition.recognizer import GestureRecognizer, Prediction

class TestGestureRecognizer(unittest.TestCase):
    def setUp(self):
        self.mock_model = Mock()
        self.actions = ["A", "B", "C"]
        self.recognizer = GestureRecognizer(
            model=self.mock_model,
            actions=self.actions,
            seq_length=3,
            threshold=0.5,
            stable_count=2
        )

    def test_init(self):
        self.assertEqual(self.recognizer.seq_length, 3)
        self.assertEqual(len(self.recognizer._actions), 3)

    def test_update_insufficient_data(self):
        # seq_length is 3. Sending 1 frame.
        fv = np.zeros(10)
        pred = self.recognizer.update(fv)
        self.assertIsNone(pred.raw_action)
        self.assertEqual(pred.confidence, 0.0)
        self.assertIsNone(pred.stable_action)

    def test_update_prediction(self):
        # Feed enough frames
        fv = np.zeros(10)

        # Setup mock model output
        # Input shape will be (1, 3, 10)
        # Output should be (3,) for one sample
        # But predict returns (batch, classes), so (1, 3)
        self.mock_model.predict.return_value = np.array([[0.1, 0.8, 0.1]]) # Class B

        self.recognizer.update(fv) # 1
        self.recognizer.update(fv) # 2
        pred = self.recognizer.update(fv) # 3 - full

        self.assertEqual(pred.raw_action, "B")
        self.assertAlmostEqual(pred.confidence, 0.8)
        self.assertIsNone(pred.stable_action) # stable_count is 2, need 2 consecutive predictions

        # Next frame, same prediction -> Stable
        pred = self.recognizer.update(fv)
        self.assertEqual(pred.raw_action, "B")
        self.assertEqual(pred.stable_action, "B")

    def test_reset(self):
        fv = np.zeros(10)
        self.mock_model.predict.return_value = np.array([[0.1, 0.8, 0.1]])

        self.recognizer.update(fv)
        self.recognizer.update(fv)
        self.recognizer.update(fv)

        # Should be full
        self.assertEqual(len(self.recognizer._seq), 3)

        self.recognizer.reset()
        self.assertEqual(len(self.recognizer._seq), 0)

        pred = self.recognizer.update(fv)
        self.assertIsNone(pred.raw_action)

    def test_low_confidence(self):
        fv = np.zeros(10)
        # High conf setup
        self.mock_model.predict.return_value = np.array([[0.1, 0.8, 0.1]])
        for _ in range(3): self.recognizer.update(fv)
        # Now seq full, next update predicts

        pred_high = self.recognizer.update(fv)
        self.assertEqual(pred_high.raw_action, "B")

        # Next frame low conf
        self.mock_model.predict.return_value = np.array([[0.4, 0.4, 0.2]]) # Max 0.4 < 0.5
        pred = self.recognizer.update(fv)

        self.assertIsNone(pred.raw_action)
        self.assertLess(pred.confidence, 0.5)
        self.assertIsNone(pred.stable_action)

if __name__ == "__main__":
    unittest.main()
