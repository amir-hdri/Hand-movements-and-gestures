import unittest
from unittest.mock import MagicMock

import numpy as np

from gesture_recognition.gui.backend.inference import SmartGestureRecognizer, InferenceEngine


class SmartRecognizerTest(unittest.TestCase):
    def setUp(self):
        self.actions = ["come", "away", "spin"]
        self.recognizer = SmartGestureRecognizer(
            model=MagicMock(),
            actions=self.actions,
            seq_length=5,
            threshold=0.8,
            stable_count=3,
            smart_thresholds={"away": 0.95},
        )

    def _feed(self, n):
        fv = np.zeros(99, dtype=np.float32)
        for _ in range(n):
            self.recognizer.update(fv)

    def test_uses_fast_call_path_not_predict(self):
        """SmartGestureRecognizer must use the direct call (tf.function) path,
        not the slow .predict() method."""
        # 0.97 is above the smart threshold (away = 0.95) so it is accepted.
        tensor = np.array([[0.1, 0.97, 0.0]], dtype=np.float32)
        model = MagicMock(return_value=tensor)
        self.recognizer._model = model

        self._feed(self.recognizer._seq_length)
        pred = self.recognizer.update(np.zeros(99, np.float32))

        model.assert_called()
        model.predict.assert_not_called()
        self.assertEqual(pred.raw_action, "away")

    def test_smart_threshold_respected(self):
        """'away' has a smart threshold of 0.95, so 0.9 must be rejected."""
        tensor = np.array([[0.1, 0.9, 0.0]], dtype=np.float32)
        self.recognizer._model = MagicMock(return_value=tensor)
        self._feed(self.recognizer._seq_length)
        pred = self.recognizer.update(np.zeros(99, np.float32))
        self.assertIsNone(pred.raw_action)
        self.assertEqual(len(self.recognizer._action_seq), 0)

    def test_low_confidence_resets_run(self):
        above = np.array([[0.1, 0.9, 0.0]], dtype=np.float32)
        below = np.array([[0.4, 0.4, 0.2]], dtype=np.float32)
        self.recognizer._model = MagicMock(return_value=above)
        self._feed(self.recognizer._seq_length)
        self.recognizer.update(np.zeros(99, np.float32))  # away
        self.recognizer.update(np.zeros(99, np.float32))  # away

        self.recognizer._model = MagicMock(return_value=below)
        pred = self.recognizer.update(np.zeros(99, np.float32))
        self.assertIsNone(pred.raw_action)
        self.assertEqual(len(self.recognizer._action_seq), 0)

    def test_shape_mismatch_returns_none_and_resets(self):
        wrong = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)  # 4 classes
        self.recognizer._model = MagicMock(return_value=wrong)
        self._feed(self.recognizer._seq_length)
        pred = self.recognizer.update(np.zeros(99, np.float32))
        self.assertIsNone(pred.raw_action)
        self.assertIsNone(pred.stable_action)
        self.assertEqual(len(self.recognizer._action_seq), 0)


if __name__ == "__main__":
    unittest.main()
