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

if __name__ == '__main__':
    unittest.main()
