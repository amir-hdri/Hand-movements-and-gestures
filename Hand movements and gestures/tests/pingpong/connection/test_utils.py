import unittest
import sys
import os
from unittest.mock import MagicMock

# Create a mock module structure for 'serial'
mock_serial = MagicMock()
mock_serial.tools = MagicMock()
mock_serial.tools.list_ports = MagicMock()

sys.modules['serial'] = mock_serial
sys.modules['serial.threaded'] = MagicMock()
sys.modules['serial.tools'] = mock_serial.tools
sys.modules['serial.tools.list_ports'] = mock_serial.tools.list_ports

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from pingpong.connection.utils import Utils

class TestUtilsFloatCheck(unittest.TestCase):

    def test_valid_float(self):
        Utils.float_check(1.5)
        Utils.float_check(-3.14)
        Utils.float_check(0.0)

    def test_valid_integer(self):
        Utils.float_check(42)
        Utils.float_check(-10)
        Utils.float_check(0)

    def test_invalid_string(self):
        with self.assertRaises(ValueError) as context:
            Utils.float_check("1.5")
        self.assertEqual(str(context.exception), "Please enter float number!")

    def test_invalid_string_with_option(self):
        with self.assertRaises(ValueError) as context:
            Utils.float_check("1.5", option="stop")
        self.assertEqual(str(context.exception), 'Please enter float number, or "stop"!')

    def test_invalid_none(self):
        with self.assertRaises(ValueError) as context:
            Utils.float_check(None)
        self.assertEqual(str(context.exception), "Please enter float number!")

    def test_invalid_none_with_option(self):
        with self.assertRaises(ValueError) as context:
            Utils.float_check(None, option="stop")
        self.assertEqual(str(context.exception), 'Please enter float number, or "stop"!')

    def test_invalid_list(self):
        with self.assertRaises(ValueError) as context:
            Utils.float_check([1.5])
        self.assertEqual(str(context.exception), "Please enter float number!")

if __name__ == '__main__':
    unittest.main()
