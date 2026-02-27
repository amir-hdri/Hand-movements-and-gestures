import sys
import re
from unittest.mock import MagicMock

# Create a mock module structure for 'serial'
mock_serial = MagicMock()
mock_serial.tools = MagicMock()
mock_serial.tools.list_ports = MagicMock()

sys.modules['serial'] = mock_serial
sys.modules['serial.threaded'] = MagicMock()
sys.modules['serial.tools'] = mock_serial.tools
sys.modules['serial.tools.list_ports'] = mock_serial.tools.list_ports

import pytest
from pingpong.operations.cube.cubeoperation import CubeOperation
from pingpong.operations.servo.servooperation import ServoOperation
from pingpong.operations.ledmatrix.ledmatrixoperation import LEDMatrixOperation
from pingpong.operations.stepper.stepperoperationbase import StepperOperationBase

class MockStatus:
    def __init__(self):
        self.sensor_prox = [10, 20, 30]
        self.connection_number = 1
        self.motor_speed = 100

class MockGroup:
    def __init__(self):
        self.processed_status = MockStatus()
        self.controller_status = MockStatus()

@pytest.fixture
def mock_robot_status():
    return {0: MockGroup(), 1: MockGroup()}

def test_code_injection_prevention(mock_robot_status):
    cube_op = CubeOperation(1, 0, mock_robot_status, lambda: None, lambda x: None)
    servo_op = ServoOperation(1, 0, mock_robot_status, lambda: None, lambda x: None)
    led_op = LEDMatrixOperation(1, 0, mock_robot_status, lambda: None, lambda x: None)
    stepper_op = StepperOperationBase(1, 0, mock_robot_status, lambda: None, lambda x: None)

    malicious_payload = '__import__("os").system("echo injected")'
    operations = [cube_op, servo_op, led_op, stepper_op]

    # Test getting non-existent malicious payload
    for op in operations:
        with pytest.raises(AttributeError):
            op._get_robot_status(0, "controller_status", malicious_payload)

    # Test setting malicious payload doesn't execute but behaves securely
    for op in operations:
        # Before fix, evaluating "__import__('os').system('echo injected') = 1" would throw SyntaxError for assignment
        # or execute the side effects depending on structure.
        # Now it simply adds it as an attribute via setattr (or it might be disallowed based on python rules)
        op._set_robot_status(0, "controller_status", malicious_payload, 1)
        # Verify it wasn't executed, just stored as literal attribute named '__import__("os").system("echo injected")'
        assert getattr(op._robot_status[0].controller_status, malicious_payload) == 1

def test_get_set_valid_indices(mock_robot_status):
    cube_op = CubeOperation(1, 0, mock_robot_status, lambda: None, lambda x: None)
    # Validate indices still work correctly and safely
    assert cube_op._get_robot_status(0, "processed_status", "sensor_prox[1]") == 20

    cube_op._set_robot_status(0, "processed_status", "sensor_prox[1]", 25)
    assert cube_op._get_robot_status(0, "processed_status", "sensor_prox[1]") == 25
    assert cube_op._robot_status[0].processed_status.sensor_prox[1] == 25
