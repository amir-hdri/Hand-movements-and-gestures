import unittest
from unittest.mock import MagicMock

import numpy as np
import tensorflow as tf

from gesture_recognition.recognizer import GestureRecognizer


class TestGestureRecognizerOptimization(unittest.TestCase):
    def test_optimized_inference(self):
        # Create a dummy model
        mock_model = MagicMock()

        # In TF 2.x, tf.function(model) returns a callable that produces a Tensor,
        # not a numpy array like predict() does.
        # We simulate the Tensor output to ensure the code handles it.
        mock_output_tensor = tf.constant([[0.1, 0.8, 0.1]], dtype=tf.float32)

        recognizer = GestureRecognizer(
            model=mock_model,
            actions=["come", "away", "spin"],
            seq_length=2,
            threshold=0.5,
            stable_count=1,
        )

        # Mock the wrapper so it doesn't actually try to run TF tracing on a MagicMock
        recognizer._model = MagicMock(return_value=mock_output_tensor)

        # Update with seq_length-1 frames
        pred1 = recognizer.update(np.zeros(99, dtype=np.float32))
        self.assertIsNone(pred1.raw_action)

        # Update with the final frame, triggering inference
        pred2 = recognizer.update(np.zeros(99, dtype=np.float32))

        # Verify model was called as a function (optimized path), not .predict()
        recognizer._model.assert_called_once()
        self.assertFalse(recognizer._model.predict.called)

        # The output tensor should have been correctly parsed
        self.assertEqual(pred2.raw_action, "away")
        self.assertAlmostEqual(pred2.confidence, 0.8, places=4)
        self.assertEqual(pred2.stable_action, "away")

if __name__ == "__main__":
    unittest.main()
