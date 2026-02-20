from abc import ABC, abstractmethod
import logging
import time
import sys
from pathlib import Path

# Add project root to sys.path to allow importing pingpong
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from pingpong.pingpong import PingPongThread
except ImportError:
    PingPongThread = None

logger = logging.getLogger(__name__)

class RobotController(ABC):
    @abstractmethod
    def connect(self):
        pass

    @abstractmethod
    def disconnect(self):
        pass

    @abstractmethod
    def send_command(self, action: str):
        pass

class MockRobotController(RobotController):
    def connect(self):
        logger.info("Mock Robot Connected")

    def disconnect(self):
        logger.info("Mock Robot Disconnected")

    def send_command(self, action: str):
        logger.info(f"Mock Robot Command: {action}")

class PingPongController(RobotController):
    def __init__(self):
        self.pingpong = None
        self.last_action = None

    def connect(self):
        if PingPongThread is None:
            logger.error("PingPong library not available")
            raise ImportError("PingPong library not available")

        try:
            self.pingpong = PingPongThread(number=2)
            self.pingpong.start()
            # self.pingpong.wait_until_full_connect() # This might block, better to check status
            logger.info("PingPong Robot Connected")
        except Exception as e:
            logger.error(f"Failed to connect to PingPong robot: {e}")
            raise

    def disconnect(self):
        if self.pingpong:
            try:
                self.pingpong.end()
            except Exception as e:
                logger.error(f"Error disconnecting PingPong robot: {e}")
            self.pingpong = None
            logger.info("PingPong Robot Disconnected")

    def send_command(self, action: str):
        if not self.pingpong:
            return

        if action == self.last_action:
            return

        self.last_action = action
        logger.info(f"Sending command to PingPong: {action}")

        # Logic from original robot.py would go here.
        # Since I don't see the specific mapping in the provided robot.py (it was commented out),
        # I will leave this as a placeholder for now or implement a basic mapping if I find one.

        # Example mapping based on action names
        # if action == "come": ...
        # elif action == "away": ...
        # elif action == "spin": ...
