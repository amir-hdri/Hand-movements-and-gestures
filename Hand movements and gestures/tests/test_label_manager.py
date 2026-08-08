import unittest
import sys
import os
from pathlib import Path
import tempfile
import shutil

# Ensure the root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# We conditionally skip this test if numpy is not available to avoid
# messing up sys.modules['numpy'] with a mock which cascades and breaks other tests.
try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

@unittest.skipUnless(HAVE_NUMPY, "Requires NumPy to test without breaking other tests in the suite")
class TestLabelManager(unittest.TestCase):
    def setUp(self):
        from gesture_recognition.gui.backend.data_manager import LabelManager
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_dir = Path(self.temp_dir)
        self.label_manager = LabelManager(self.dataset_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_get_label_index_missing_label(self):
        with self.assertRaises(ValueError) as context:
            self.label_manager.get_label_index("non_existent_label")
        self.assertEqual(str(context.exception), "Label non_existent_label not found")

    def test_add_label_rejects_empty(self):
        with self.assertRaises(ValueError):
            self.label_manager.add_label("   ")

    def test_add_label_rejects_path_separators(self):
        with self.assertRaises(ValueError):
            self.label_manager.add_label("a/b")
        with self.assertRaises(ValueError):
            self.label_manager.add_label("a\\b")

    def test_add_label_strips_and_adds(self):
        self.label_manager.add_label("  swipe  ")
        self.assertIn("swipe", self.label_manager.get_labels())

    def test_remove_gesture_only_removes_exact_label(self):
        """Deleting 'spin' must not delete 'spin2' or 'left_spin' data."""
        from unittest.mock import patch
        import numpy as np
        from gesture_recognition.gui.backend.data_manager import DataManager

        manager = DataManager()
        # Isolate the dataset dir and label manager.
        dataset_dir = Path(self.temp_dir)
        with patch("gesture_recognition.gui.backend.data_manager.config") as mock_cfg:
            mock_cfg.DATASET_DIR = dataset_dir
            mock_cfg.SEQ_LENGTH = 30
            manager.config = mock_cfg

            # Create files for the exact label and near-miss labels.
            files = [
                "raw_spin_1.npy", "seq_spin_1.npy",
                "raw_spin2_1.npy", "seq_spin2_1.npy",
                "raw_left_spin_1.npy", "seq_left_spin_1.npy",
            ]
            for name in files:
                np.save(dataset_dir / name, np.zeros((1, 99), dtype=np.float32))

            manager.label_manager = self.label_manager
            self.label_manager.add_label("spin")
            self.label_manager.add_label("spin2")
            self.label_manager.add_label("left_spin")

            manager.remove_gesture("spin")

            self.assertFalse((dataset_dir / "raw_spin_1.npy").exists())
            self.assertFalse((dataset_dir / "seq_spin_1.npy").exists())
            # Near-miss labels must survive.
            self.assertTrue((dataset_dir / "raw_spin2_1.npy").exists())
            self.assertTrue((dataset_dir / "seq_spin2_1.npy").exists())
            self.assertTrue((dataset_dir / "raw_left_spin_1.npy").exists())
            self.assertTrue((dataset_dir / "seq_left_spin_1.npy").exists())

if __name__ == '__main__':
    unittest.main()
