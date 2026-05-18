import unittest
import sys
import os
import unittest.mock

# Mock serial before importing any pingpong module
sys.modules['serial'] = unittest.mock.MagicMock()
sys.modules['serial.tools'] = unittest.mock.MagicMock()
sys.modules['serial.tools.list_ports'] = unittest.mock.MagicMock()

# Add the project root to sys.path if not running through a proper runner
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from pingpong.connection.utils import Utils

class TestUtils(unittest.TestCase):
    def test_bytes_to_hex_str_basic(self):
        """Test formatting of a standard byte array"""
        self.assertEqual(Utils.bytes_to_hex_str(b'\x01\x02\x0a\xff'), "01 02 0A FF")

    def test_bytes_to_hex_str_single_byte(self):
        """Test formatting of a single byte"""
        self.assertEqual(Utils.bytes_to_hex_str(b'\x00'), "00")

    def test_bytes_to_hex_str_empty(self):
        """Test formatting of empty bytes"""
        self.assertEqual(Utils.bytes_to_hex_str(b''), "")

    def test_bytes_to_hex_str_all_zeros(self):
        """Test formatting of a byte array containing only zeros"""
        self.assertEqual(Utils.bytes_to_hex_str(b'\x00\x00\x00\x00'), "00 00 00 00")

    def test_bytes_to_hex_str_large_array(self):
        """Test formatting of a larger byte array"""
        large_bytes = bytes(range(256))
        expected_hex = " ".join(f"{b:02X}" for b in range(256))
        self.assertEqual(Utils.bytes_to_hex_str(large_bytes), expected_hex)

    def test_bytes_to_hex_str_mixed_case(self):
        """Test formatting of bytes that result in mixed case hex strings natively"""
        # Ensure that characters like a-f are uppercased to A-F
        self.assertEqual(Utils.bytes_to_hex_str(b'\xab\xcd\xef'), "AB CD EF")

if __name__ == '__main__':
    unittest.main()
