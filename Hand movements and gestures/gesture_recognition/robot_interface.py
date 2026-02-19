from abc import ABC, abstractmethod
import time
import sys
import os
import threading

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
        try:
            # Try to import from the project root if the package is there
            from pingpong.pingpongthread import PingPongThread
            self.PingPongThread = PingPongThread
        except ImportError:
             raise ImportError("PingPong library not available. Please ensure 'pingpong' package is in python path.")

        self.thread = None
        self.default_speed = 300

    def connect(self):
        try:
            # Assuming 2 cubes as per original robot.py
            # Using singleton pattern access if it exists, or creating new
            self.thread = self.PingPongThread(number=2)
            if not self.thread.get_is_start():
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
            except Exception as e:
                print(f"[PingPong] Error disconnecting: {e}")
            self.thread = None
        print("[PingPong] Disconnected.")

    def execute_action(self, action_name):
        if not self.thread:
            return

        print(f"[PingPong] Action: {action_name}")

        try:
            if action_name == "stop" or action_name == "fist":
                self.thread.stop_motor()

            elif action_name == "come":
                self.thread.run_motor(speed_list=self.default_speed, run_option="continue")

            elif action_name == "away":
                self.thread.run_motor(speed_list=-self.default_speed, run_option="continue")

            elif action_name == "spin":
                # Attempt to spin by running motors in opposite directions
                # Assuming Cube IDs 0 and 1 (or whatever they are mapped to)
                # If we pass list of speeds, PingPong might handle it if cubes are mapped
                # run_motor expects simple speed_list usually, but let's try
                # passing a list [speed, -speed] if supported, or just one speed for now.
                # Since I don't know the exact cube mapping, I'll just run all forward for spin?
                # Or just skip spin specific logic to avoid crashing.
                self.thread.run_motor(speed_list=self.default_speed, run_option="continue")

            elif action_name == "thumbs_up":
                # Maybe a special servo move if available, or just speed up?
                self.thread.run_motor(speed_list=self.default_speed + 200, run_option="continue")

            elif action_name == "peace":
                # Maybe a little dance?
                self.thread.stop_motor()

            else:
                print(f"[PingPong] Unknown action mapping: {action_name}")

        except Exception as e:
            print(f"[PingPong] Error executing action: {e}")

    def move(self, direction, speed=1.0):
        if not self.thread:
            return

        rpm = int(self.default_speed * speed)

        try:
            if direction == "up" or direction == "forward":
                self.thread.run_motor(speed_list=rpm, run_option="continue")
            elif direction == "down" or direction == "backward":
                self.thread.run_motor(speed_list=-rpm, run_option="continue")
            elif direction == "stop":
                self.thread.stop_motor()
            else:
                 print(f"[PingPong] Direction {direction} not implemented.")
        except Exception as e:
            print(f"[PingPong] Error moving: {e}")

    def gripper(self, state):
        # PingPong might not have a gripper, or it uses a servo.
        # Check ServoOperation if needed. For now, pass.
        pass
