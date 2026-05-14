import unittest

from pingpong.connection.utils import Utils

class TestUtils(unittest.TestCase):
    def test_float_check_valid_float_within_range(self):
        # Happy path - value within range
        self.assertEqual(Utils.float_check(5.0, 0.0, 10.0), 5.0)
        self.assertEqual(Utils.float_check(0.0, -5.0, 5.0), 0.0)

    def test_float_check_valid_int_within_range(self):
        # Happy path with int - should return float
        self.assertEqual(Utils.float_check(5, 0.0, 10.0), 5.0)
        self.assertEqual(Utils.float_check(0, -5.0, 5.0), 0.0)

    def test_float_check_below_min(self):
        # Value below min should return min
        self.assertEqual(Utils.float_check(-1.0, 0.0, 10.0), 0.0)
        self.assertEqual(Utils.float_check(-5, 0.0, 10.0), 0.0)

    def test_float_check_above_max(self):
        # Value above max should return max
        self.assertEqual(Utils.float_check(11.0, 0.0, 10.0), 10.0)
        self.assertEqual(Utils.float_check(15, 0.0, 10.0), 10.0)

    def test_float_check_invalid_type(self):
        # Non-float/int types should return -1
        self.assertEqual(Utils.float_check("5.0", 0.0, 10.0), -1)
        self.assertEqual(Utils.float_check([5.0], 0.0, 10.0), -1)
        self.assertEqual(Utils.float_check(None, 0.0, 10.0), -1)

    def test_float_check_boolean(self):
        # type(True) is bool, not int or float, so it should return -1
        self.assertEqual(Utils.float_check(True, 0.0, 10.0), -1)
        self.assertEqual(Utils.float_check(False, 0.0, 10.0), -1)

    def test_float_check_edge_cases(self):
        # Value exactly at min
        self.assertEqual(Utils.float_check(0.0, 0.0, 10.0), 0.0)
        # Value exactly at max
        self.assertEqual(Utils.float_check(10.0, 0.0, 10.0), 10.0)

if __name__ == "__main__":
    unittest.main()
