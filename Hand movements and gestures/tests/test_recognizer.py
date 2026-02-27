import unittest
from unittest.mock import Mock

import numpy as np

from gesture_recognition.recognizer import GestureRecognizer, Prediction


class RecognizerTest(unittest.TestCase):
    def setUp(self):
        self.actions = ["action1", "action2", "action3"]
        self.mock_model = Mock()
        self.seq_length = 5
        self.threshold = 0.8
        self.stable_count = 3

        self.recognizer = GestureRecognizer(
            model=self.mock_model,
            actions=self.actions,
            seq_length=self.seq_length,
            threshold=self.threshold,
            stable_count=self.stable_count,
        )

    def test_init_invalid_args(self):
        with self.assertRaises(ValueError):
            GestureRecognizer(self.mock_model, self.actions, seq_length=0)

        with self.assertRaises(ValueError):
            GestureRecognizer(self.mock_model, self.actions, stable_count=-1)

    def test_update_insufficient_sequence(self):
        feature_vector = np.zeros(99, dtype=np.float32)

        # Test updating with less than seq_length
        for i in range(self.seq_length - 1):
            pred = self.recognizer.update(feature_vector)
            self.assertIsNone(pred.raw_action)
            self.assertEqual(pred.confidence, 0.0)
            self.assertIsNone(pred.stable_action)

        self.mock_model.predict.assert_not_called()

    def test_update_below_threshold(self):
        # Model returns predictions where max confidence is below threshold
        # For a sequence of enough length
        feature_vector = np.zeros(99, dtype=np.float32)

        # Mock prediction returning array of shape (1, 3) where max is 0.7
        mock_pred = np.array([[0.1, 0.7, 0.2]], dtype=np.float32)
        self.mock_model.predict.return_value = mock_pred

        # Feed enough to trigger prediction
        for _ in range(self.seq_length):
            pred = self.recognizer.update(feature_vector)

        self.assertIsNone(pred.raw_action)
        self.assertAlmostEqual(pred.confidence, 0.7)
        self.assertIsNone(pred.stable_action)

    def test_update_above_threshold_stable_action(self):
        feature_vector = np.zeros(99, dtype=np.float32)

        # Mock prediction returning max confidence > threshold for "action2"
        mock_pred = np.array([[0.1, 0.9, 0.0]], dtype=np.float32)
        self.mock_model.predict.return_value = mock_pred

        # First seq_length-1 inputs: no prediction
        for _ in range(self.seq_length - 1):
            self.recognizer.update(feature_vector)

        # First prediction
        pred1 = self.recognizer.update(feature_vector)
        self.assertEqual(pred1.raw_action, "action2")
        self.assertAlmostEqual(pred1.confidence, 0.9)
        self.assertIsNone(pred1.stable_action) # stable_count is 3, only 1 so far

        # Second prediction
        pred2 = self.recognizer.update(feature_vector)
        self.assertEqual(pred2.raw_action, "action2")
        self.assertAlmostEqual(pred2.confidence, 0.9)
        self.assertIsNone(pred2.stable_action)

        # Third prediction (stable_count reached)
        pred3 = self.recognizer.update(feature_vector)
        self.assertEqual(pred3.raw_action, "action2")
        self.assertAlmostEqual(pred3.confidence, 0.9)
        self.assertEqual(pred3.stable_action, "action2")

    def test_update_unexpected_model_output_shape(self):
        feature_vector = np.zeros(99, dtype=np.float32)

        # Model returns mismatched action count (4 instead of 3)
        mock_pred = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
        self.mock_model.predict.return_value = mock_pred

        # Fill sequence length - 1
        for _ in range(self.seq_length - 1):
            self.recognizer.update(feature_vector)

        with self.assertRaises(ValueError) as context:
            self.recognizer.update(feature_vector)

        self.assertIn("Unexpected model output shape", str(context.exception))

    def test_update_typeerror_fallback_verbose(self):
        feature_vector = np.zeros(99, dtype=np.float32)

        def mock_predict(*args, **kwargs):
            if 'verbose' in kwargs:
                raise TypeError("predict() got an unexpected keyword argument 'verbose'")
            return np.array([[0.1, 0.85, 0.05]], dtype=np.float32)

        self.mock_model.predict.side_effect = mock_predict

        # Fill up sequence to trigger prediction
        for _ in range(self.seq_length):
            pred = self.recognizer.update(feature_vector)

        self.assertEqual(pred.raw_action, "action2")
        self.assertAlmostEqual(pred.confidence, 0.85)

if __name__ == "__main__":
    unittest.main()
