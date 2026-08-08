import unittest
import os
from importlib import reload


class ConfigEnvTest(unittest.TestCase):
    def tearDown(self):
        for var in ("SEQ_LENGTH", "CONFIDENCE_THRESHOLD", "STABLE_COUNT", "CORS_ALLOWED_ORIGINS"):
            os.environ.pop(var, None)

    def _reload(self):
        import gesture_recognition.gui.backend.config as cfg

        reload(cfg)
        return cfg

    def test_defaults(self):
        cfg = self._reload()
        self.assertEqual(cfg.config.SEQ_LENGTH, 30)
        self.assertEqual(cfg.config.DEFAULT_THRESHOLD, 0.9)
        self.assertEqual(cfg.config.STABLE_COUNT, 3)

    def test_env_overrides(self):
        os.environ["SEQ_LENGTH"] = "40"
        os.environ["CONFIDENCE_THRESHOLD"] = "0.75"
        os.environ["STABLE_COUNT"] = "5"
        cfg = self._reload()
        self.assertEqual(cfg.config.SEQ_LENGTH, 40)
        self.assertEqual(cfg.config.DEFAULT_THRESHOLD, 0.75)
        self.assertEqual(cfg.config.STABLE_COUNT, 5)

    def test_env_invalid_values_fall_back(self):
        os.environ["SEQ_LENGTH"] = "not-a-number"
        os.environ["CONFIDENCE_THRESHOLD"] = ""
        cfg = self._reload()
        self.assertEqual(cfg.config.SEQ_LENGTH, 30)
        self.assertEqual(cfg.config.DEFAULT_THRESHOLD, 0.9)

    def test_get_threshold_falls_back_to_default(self):
        cfg = self._reload()
        self.assertEqual(cfg.config.get_threshold("unknown"), cfg.config.DEFAULT_THRESHOLD)
        self.assertEqual(cfg.config.get_threshold("stop"), 0.98)


if __name__ == "__main__":
    unittest.main()
