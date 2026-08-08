import unittest


class AppStateHistoryTest(unittest.TestCase):
    """Tests for AppState.add_prediction_to_history deduplication."""

    @classmethod
    def setUpClass(cls):
        # Import lazily: importing the module starts a (daemon) camera thread
        # and loads the model, so we only pay that cost for this test class.
        from gesture_recognition.gui.backend import app as app_module

        cls.AppState = app_module.AppState

    def setUp(self):
        # Build an AppState without running the heavy __init__ (model load).
        self.state = self.AppState.__new__(self.AppState)
        import threading

        self.state.history_lock = threading.Lock()
        self.state.prediction_history = []
        self.state.last_history_action = None
        self.state.last_history_confidence = 0.0

    def test_consecutive_duplicates_collapsed(self):
        self.state.add_prediction_to_history("away", 0.95)
        self.state.add_prediction_to_history("away", 0.96)
        self.state.add_prediction_to_history("away", 0.94)
        self.assertEqual(len(self.state.prediction_history), 1)
        self.assertEqual(self.state.prediction_history[0]["action"], "away")

    def test_action_change_adds_entry(self):
        self.state.add_prediction_to_history("away", 0.95)
        self.state.add_prediction_to_history("come", 0.9)
        self.assertEqual(len(self.state.prediction_history), 2)
        self.assertEqual(self.state.prediction_history[-1]["action"], "come")

    def test_none_is_ignored(self):
        self.state.add_prediction_to_history(None, 0.0)
        self.assertEqual(len(self.state.prediction_history), 0)

    def test_clear_resets_tracking(self):
        self.state.add_prediction_to_history("away", 0.95)
        self.state.clear_prediction_history()
        self.state.add_prediction_to_history("away", 0.95)
        self.assertEqual(len(self.state.prediction_history), 1)

    def test_history_capped_at_100(self):
        for i in range(150):
            self.state.add_prediction_to_history(f"g{i % 3}", 0.9)
        # Alternate actions so every call adds an entry, then verify the cap.
        self.state.prediction_history = []
        self.state.last_history_action = None
        actions = ["a", "b", "c"] * 60
        for a in actions:
            self.state.add_prediction_to_history(a, 0.9)
        self.assertEqual(len(self.state.prediction_history), 100)


if __name__ == "__main__":
    unittest.main()
