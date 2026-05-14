import unittest
from unittest.mock import MagicMock, patch
import sys

class TestLabelManager(unittest.TestCase):

    @patch.dict('sys.modules', {
        'mediapipe': MagicMock(),
        'mediapipe.tasks': MagicMock(),
        'mediapipe.tasks.python': MagicMock(),
        'mediapipe.tasks.python.vision': MagicMock(),
        'cv2': MagicMock()
    })
    def test_get_label_index_error(self):
        from pathlib import Path
        from gesture_recognition.gui.backend.data_manager import LabelManager

        mock_dataset_dir = MagicMock(spec=Path)
        mock_labels_file = MagicMock(spec=Path)
        mock_dataset_dir.__truediv__.return_value = mock_labels_file
        mock_labels_file.exists.return_value = False

        with patch("gesture_recognition.gui.backend.data_manager.config") as mock_config:
            mock_config.ACTIONS = ["action1", "action2"]
            manager = LabelManager(mock_dataset_dir)

            with self.assertRaises(ValueError) as context:
                manager.get_label_index("nonexistent_action")

            self.assertEqual(str(context.exception), "Label nonexistent_action not found")

if __name__ == "__main__":
    unittest.main()
