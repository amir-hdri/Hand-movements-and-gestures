import unittest
import sys
from unittest.mock import MagicMock

# Mock serial and numpy modules for tests since we're offline
sys.modules['serial'] = MagicMock()
sys.modules['serial.tools'] = MagicMock()
sys.modules['serial.tools.list_ports'] = MagicMock()

try:
    from pingpong.operations.ledmatrix.ledmatrixoperation import LEDMatrixOperation
except ImportError:
    pass # Assume run from root

class DummyControllerStatus:
    def __init__(self, connection_number=1, group_id=0):
        self.connection_number = connection_number
        self.stepper_mode = [None] * connection_number
        self.some_var = 0

class DummyRobotStatus:
    def __init__(self, connection_number=1):
        self.controller_status = DummyControllerStatus(connection_number)

class TestLEDMatrixOperation(unittest.TestCase):
    def setUp(self):
        # We try importing locally to match the test runner environment
        try:
            from pingpong.operations.ledmatrix.ledmatrixoperation import LEDMatrixOperation
            self.LEDMatrixOperation = LEDMatrixOperation
        except ImportError:
            # Maybe the path is not set up correctly in the test runner
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
            from pingpong.operations.ledmatrix.ledmatrixoperation import LEDMatrixOperation
            self.LEDMatrixOperation = LEDMatrixOperation

    def test_set_robot_status_simple_attr(self):
        """Test setting a simple attribute securely without exec()."""
        group_id = 0
        robot_status = {group_id: DummyRobotStatus()}

        # We need to mock start_check and write
        start_check = MagicMock()
        write = MagicMock()

        # Use mocked GenerateProtocol
        with unittest.mock.patch('pingpong.operations.ledmatrix.ledmatrixoperation.GenerateProtocol'):
            op = self.LEDMatrixOperation(1, group_id, robot_status, start_check, write)
            op._set_robot_status(group_id, "controller_status", "some_var", 42)
            self.assertEqual(robot_status[group_id].controller_status.some_var, 42)

    def test_set_robot_status_indexed_attr(self):
        """Test setting an indexed attribute like stepper_mode[0] securely."""
        group_id = 1
        connection_number = 2
        robot_status = {group_id: DummyRobotStatus(connection_number)}

        start_check = MagicMock()
        write = MagicMock()

        with unittest.mock.patch('pingpong.operations.ledmatrix.ledmatrixoperation.GenerateProtocol'):
            op = self.LEDMatrixOperation(1, group_id, robot_status, start_check, write)
            op._set_robot_status(group_id, "controller_status", "stepper_mode[1]", "test_mode")
            self.assertEqual(robot_status[group_id].controller_status.stepper_mode[1], "test_mode")

if __name__ == "__main__":
    unittest.main()
