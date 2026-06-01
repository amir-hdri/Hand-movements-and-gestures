import unittest
import sys
from unittest.mock import MagicMock

# Mock serial module
mock_serial = MagicMock()
sys.modules['serial'] = mock_serial
sys.modules['serial.tools'] = MagicMock()
sys.modules['serial.tools.list_ports'] = MagicMock()

# Now we can import Utils
from pingpong.connection.utils import Utils

class TestUtils(unittest.TestCase):
    def setUp(self):
        self.utils = Utils()

    def test_list_product_copy_basic(self):
        input_list = [[1, 2, 3]]
        number = 2
        expected = [[1, 2, 3], [1, 2, 3]]
        result = Utils.list_product_copy(input_list, number)
        self.assertEqual(result, expected)

    def test_list_product_copy_independence(self):
        input_list = [[1, 2, 3]]
        number = 2
        result = Utils.list_product_copy(input_list, number)
        # Modify the first element of the first sub-list
        result[0][0] = 99
        self.assertEqual(result[0], [99, 2, 3])
        self.assertEqual(result[1], [1, 2, 3])

    def test_list_product_copy_zero(self):
        input_list = [[1]]
        number = 0
        expected = []
        result = Utils.list_product_copy(input_list, number)
        self.assertEqual(result, expected)

    def test_bytes_to_hex_str(self):
        self.assertEqual(Utils.bytes_to_hex_str(b'\x01\x02\x0f\x10'), "01 02 0F 10")
        self.assertEqual(Utils.bytes_to_hex_str(b''), "")

    def test_float_check(self):
        # Should not raise
        Utils.float_check(1.0)
        Utils.float_check(1)

        # Should raise
        with self.assertRaises(ValueError) as cm:
            Utils.float_check("1")
        self.assertEqual(str(cm.exception), "Please enter float number!")

        with self.assertRaises(ValueError) as cm:
            Utils.float_check("1", option="test_opt")
        self.assertEqual(str(cm.exception), 'Please enter float number, or "test_opt"!')

    def test_integer_check(self):
        # Should not raise
        Utils.integer_check(1)

        # Should raise
        with self.assertRaises(ValueError) as cm:
            Utils.integer_check(1.0)
        self.assertEqual(str(cm.exception), "Please enter integer number!")

        with self.assertRaises(ValueError) as cm:
            Utils.integer_check("1", option="test_opt")
        self.assertEqual(str(cm.exception), 'Please enter integer number, or "test_opt"!')

    def test_twobyte_hexlist_to_int(self):
        self.assertEqual(Utils.twobyte_hexlist_to_int(0x12, 0x34), 4660)
        # Verify correct padding/shifting for values < 16
        self.assertEqual(Utils.twobyte_hexlist_to_int(0x01, 0x02), 258)

    def test_to_list(self):
        self.assertEqual(Utils.to_list(1), [1])
        self.assertEqual(Utils.to_list("a"), ["a"])
        self.assertEqual(Utils.to_list(None), [None])
        self.assertEqual(Utils.to_list([1, 2]), [1, 2])
        self.assertEqual(Utils.to_list((1, 2)), [1, 2])
        with self.assertRaises(ValueError):
            Utils.to_list({})

    def test_check_same_element(self):
        # Should not raise
        Utils.check_same_element([1, 2, 3])

        # Should raise
        with self.assertRaises(ValueError):
            Utils.check_same_element([1, 2, 1])

    def test_all_cube_in_check(self):
        self.assertTrue(Utils.all_cube_in_check([0, 1, 2], 3))
        self.assertFalse(Utils.all_cube_in_check([0, 2], 3))

    def test_check_group_id(self):
        self.assertEqual(Utils.check_group_id(1), 1)
        self.assertEqual(Utils.check_group_id("1"), 1)
        self.assertIsNone(Utils.check_group_id(None))

        # Excluded IDs
        for not_id in (11, 22, 33, 44, 55, 66):
            with self.assertRaises(ValueError):
                Utils.check_group_id(not_id)

        # Out of range
        with self.assertRaises(ValueError):
            Utils.check_group_id(0)
        with self.assertRaises(ValueError):
            Utils.check_group_id(77)

        # Invalid type
        with self.assertRaises(ValueError):
            Utils.check_group_id([])

    def test_getSignedIntfromByteData(self):
        self.assertEqual(self.utils.getSignedIntfromByteData(0), 0)
        self.assertEqual(self.utils.getSignedIntfromByteData(127), 127)
        self.assertEqual(self.utils.getSignedIntfromByteData(128), -128)
        self.assertEqual(self.utils.getSignedIntfromByteData(255), -1)

    def test_getACCDataToDegreeMinus90To90fromByteData(self):
        self.assertEqual(self.utils.getACCDataToDegreeMinus90To90fromByteData(0), 0)
        self.assertEqual(self.utils.getACCDataToDegreeMinus90To90fromByteData(45), 45)
        self.assertEqual(self.utils.getACCDataToDegreeMinus90To90fromByteData(90), 90)
        self.assertEqual(self.utils.getACCDataToDegreeMinus90To90fromByteData(100), 90)
        self.assertEqual(self.utils.getACCDataToDegreeMinus90To90fromByteData(255), -1)
        self.assertEqual(self.utils.getACCDataToDegreeMinus90To90fromByteData(200), -56) # 200 - 256 = -56
        self.assertEqual(self.utils.getACCDataToDegreeMinus90To90fromByteData(150), -90) # 150 - 256 = -106 -> clamped to -90

if __name__ == "__main__":
    unittest.main()
