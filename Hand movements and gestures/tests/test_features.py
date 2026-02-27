import unittest
import sys

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    import unittest.mock
    HAVE_NUMPY = False
    sys.modules['numpy'] = unittest.mock.MagicMock()
    np = sys.modules['numpy']

try:
    from gesture_recognition.features import (
        append_label,
        hand_landmarks_to_feature_vector,
        hand_landmarks_to_joint,
        joint_to_angles,
    )
except ImportError:
    pass

class _LM:
    def __init__(self, x: float, y: float, z: float, visibility: float | None = None) -> None:
        self.x = x
        self.y = y
        self.z = z
        if visibility is not None:
            self.visibility = visibility


class _Hand:
    def __init__(self, landmarks):
        self.landmark = landmarks

@unittest.skipUnless(HAVE_NUMPY, "Requires NumPy")
class FeaturesTest(unittest.TestCase):
    def test_joint_shape_and_visibility_default(self):
        hand = _Hand([_LM(i, i + 1, i + 2) for i in range(21)])
        joint = hand_landmarks_to_joint(hand)
        self.assertEqual(joint.shape, (21, 4))
        self.assertEqual(joint.dtype, np.float32)
        self.assertTrue(np.all(joint[:, 3] == 0.0))

    def test_feature_vector_and_label(self):
        hand = _Hand([_LM(i, i + 1, i + 2, visibility=0.5) for i in range(21)])
        fv = hand_landmarks_to_feature_vector(hand)
        self.assertEqual(fv.shape, (99,))
        self.assertEqual(fv.dtype, np.float32)

        with_label = append_label(fv, 2)
        self.assertEqual(with_label.shape, (100,))
        self.assertEqual(with_label.dtype, np.float32)
        self.assertEqual(with_label[-1], 2.0)

    def test_angles_no_nan_for_zero_vectors(self):
        # All-zero joints => zero vectors => should not produce NaNs (we clamp norms with eps).
        joint = np.zeros((21, 4), dtype=np.float32)
        angles = joint_to_angles(joint)
        self.assertEqual(angles.shape, (15,))
        self.assertEqual(angles.dtype, np.float32)
        self.assertFalse(np.isnan(angles).any())
        # dot(0,0)=0 => arccos(0)=90deg
        self.assertTrue(np.allclose(angles, 90.0, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
