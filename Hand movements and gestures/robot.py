import argparse
import sys
import time
from pathlib import Path
import cv2

# Ensure project root is in path
sys.path.append(str(Path(__file__).resolve().parent))

from gesture_recognition.config import GestureConfig
from gesture_recognition.pipeline import GesturePipeline
from gesture_recognition.utils import setup_logging, open_camera
from gesture_recognition.robot_interface import PingPongController, MockRobotController

logger = setup_logging(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gesture control loop for PingPong Robot.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    # Config will load defaults, but CLI can override if needed (not fully implemented here, relying on config.py mostly)
    # But model path is useful
    parser.add_argument("--model", type=Path, default=None, help="Path to model file")
    parser.add_argument("--enable-robot", action="store_true", help="Enable PingPong robot")
    parser.add_argument("--mock-robot", action="store_true", help="Use Mock robot if real one unavailable")
    return parser.parse_args()

def main():
    args = parse_args()

    # Load default config
    config = GestureConfig()

    # Override config with args if provided
    if args.model:
        config.model_path = args.model

    # Override camera
    config.camera_id = args.camera

    logger.info(f"Configuration: {config}")

    robot = None
    if args.enable_robot:
        try:
            logger.info("Initializing PingPong robot...")
            robot = PingPongController()
            if not robot.connect():
                logger.error("Failed to connect to PingPong robot.")
                robot = None
            else:
                logger.info("PingPong connected.")
        except ImportError:
            logger.error("PingPong library not available.")
            if args.mock_robot:
                logger.info("Falling back to Mock Robot.")
                robot = MockRobotController()
                robot.connect()
    elif args.mock_robot:
        logger.info("Using Mock Robot.")
        robot = MockRobotController()
        robot.connect()

    pipeline = GesturePipeline(config)

    try:
        cap = open_camera(config.camera_id)
    except Exception as e:
        logger.error(f"Failed to open camera: {e}")
        pipeline.close()
        if robot: robot.disconnect()
        return

    last_action = None
    last_action_time = 0

    try:
        logger.info("Starting robot control loop. Press 'q' to exit.")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame, prediction = pipeline.process_frame(frame)

            if prediction and prediction.stable_action:
                this_action = prediction.stable_action

                # Execute action only if it changed OR if enough time passed (debounce)
                # But typically we only send command on change for state-based control,
                # or periodically for continuous control.
                # Here we send on change.

                if robot and last_action != this_action:
                    logger.info(f"Stable Action detected: {this_action}")
                    robot.execute_action(this_action)
                    last_action = this_action
                    last_action_time = time.time()

                # Optional: Send "continue" command every second if action is "come" or "away"
                # to keep robot moving if it has a timeout.
                elif robot and (time.time() - last_action_time > 1.0) and this_action in ["come", "away"]:
                     robot.execute_action(this_action)
                     last_action_time = time.time()


            cv2.imshow("Gesture Robot Control", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pipeline.close()
        if robot:
            robot.disconnect()

if __name__ == "__main__":
    main()
