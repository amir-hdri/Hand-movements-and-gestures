import argparse
import sys
from pathlib import Path
import cv2

# Ensure project root is in path
sys.path.append(str(Path(__file__).resolve().parent))

from gesture_recognition.config import GestureConfig
from gesture_recognition.pipeline import GesturePipeline
from gesture_recognition.utils import setup_logging, open_camera

logger = setup_logging(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gesture control loop for PingPong Robot.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--model", type=Path, default=Path("models/model2_1.0.h5"))
    parser.add_argument("--actions", nargs="+", default=["come", "away", "spin"])
    parser.add_argument("--seq-length", type=int, default=30)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--stable-count", type=int, default=3)
    parser.add_argument("--enable-robot", action="store_true", help="Enable PingPong robot")
    return parser.parse_args()

def _ensure_pingpong_on_path() -> None:
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

def main():
    args = parse_args()

    config = GestureConfig(
        camera_id=args.camera,
        model_path=args.model,
        actions=args.actions,
        seq_length=args.seq_length,
        threshold=args.threshold,
        stable_count=args.stable_count,
        save_video=False
    )

    pingpong = None
    if args.enable_robot:
        _ensure_pingpong_on_path()
        try:
            from pingpong import PingPongThread
        except Exception as exc:
            logger.error(f"PingPong library not available: {exc}")
            return

        logger.info("Initializing PingPong robot...")
        pingpong = PingPongThread(number=2)
        pingpong.start()
        pingpong.wait_until_full_connect()
        logger.info("PingPong connected.")

    pipeline = GesturePipeline(config)

    try:
        cap = open_camera(config.camera_id)
    except Exception as e:
        logger.error(f"Failed to open camera: {e}")
        return

    last_action = None

    try:
        logger.info("Starting robot control loop. Press 'q' to exit.")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame, prediction = pipeline.process_frame(frame)

            if prediction and prediction.stable_action:
                this_action = prediction.stable_action
                if pingpong and last_action != this_action:
                    logger.info(f"Stable Action detected: {this_action}")
                    # Map gestures to robot actions here.
                    # Example:
                    # if this_action == "come": pingpong.action_come()
                    last_action = this_action

            cv2.imshow("Gesture Robot Control", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pipeline.close()
        if pingpong:
            try:
                pingpong.end()
            except Exception:
                pass

if __name__ == "__main__":
    main()
