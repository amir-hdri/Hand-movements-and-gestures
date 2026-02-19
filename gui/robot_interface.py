from abc import ABC, abstractmethod
import sys
import os

# Try to import PingPong only if available
try:
    # Add project root to path to find pingpong module
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Hand movements and gestures"))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from pingpong.pingpongthread import PingPongThread
    PINGPONG_AVAILABLE = True
except ImportError:
    PINGPONG_AVAILABLE = False
except Exception as e:
    print(f"Error importing PingPong: {e}")
    PINGPONG_AVAILABLE = False


class RobotController(ABC):
    """Abstract base class for robot controllers."""

    @abstractmethod
    def connect(self):
        """Connect to the robot."""
        pass

    @abstractmethod
    def disconnect(self):
        """Disconnect from the robot."""
        pass

    @abstractmethod
    def execute_action(self, action_name):
        """Execute a specific named action (e.g., from gesture recognition)."""
        pass

    # Manual controls
    @abstractmethod
    def move(self, direction, speed=1.0):
        """Move the robot in a direction (up, down, left, right, forward, backward)."""
        pass

    @abstractmethod
    def gripper(self, state):
        """Control gripper (open, close)."""
        pass


class MockRobotController(RobotController):
    """A mock controller that prints actions to stdout."""

    def __init__(self, name="Mock Robot"):
        self.name = name
        self.connected = False

    def connect(self):
        self.connected = True
        print(f"[{self.name}] Connected.")
        return True

    def disconnect(self):
        self.connected = False
        print(f"[{self.name}] Disconnected.")

    def execute_action(self, action_name):
        if not self.connected:
            print(f"[{self.name}] Not connected. Cannot execute '{action_name}'.")
            return
        print(f"[{self.name}] Executing action: {action_name}")

    def move(self, direction, speed=1.0):
        if not self.connected:
            return
        print(f"[{self.name}] Moving {direction} at speed {speed}")

    def gripper(self, state):
        if not self.connected:
            return
        print(f"[{self.name}] Gripper: {state}")


class PingPongController(RobotController):
    """Controller for the PingPong robot."""

    def __init__(self):
        if not PINGPONG_AVAILABLE:
            raise ImportError("PingPong library not available.")
        self.thread = None

    def connect(self):
        try:
            self.thread = PingPongThread(number=2) # Defaulting to 2 as in original code
            self.thread.start()
            self.thread.wait_until_full_connect()
            print("[PingPong] Connected.")
            return True
        except Exception as e:
            print(f"[PingPong] Connection failed: {e}")
            return False

    def disconnect(self):
        if self.thread:
            try:
                self.thread.end()
            except:
                pass
            self.thread = None
        print("[PingPong] Disconnected.")

    def execute_action(self, action_name):
        if not self.thread:
            return
        # TODO: Map actions to PingPong specific commands
        # For now just print
        print(f"[PingPong] Received action: {action_name}")

    def move(self, direction, speed=1.0):
        if not self.thread:
            return
        print(f"[PingPong] Move {direction} not implemented yet.")

    def gripper(self, state):
        pass
