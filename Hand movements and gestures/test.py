import argparse
from pathlib import Path
import cv2
import sys

# Ensure project root is in path
sys.path.append(str(Path(__file__).resolve().parent))

from gesture_recognition.config import GestureConfig
from gesture_recognition.pipeline import GesturePipeline
from gesture_recognition.utils import setup_logging, open_camera

logger = setup_logging(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime gesture recognition demo.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--model", type=Path, default=Path("models/model2_1.0.h5"))
    parser.add_argument("--actions", nargs="+", default=["come", "away", "spin"])
    parser.add_argument("--seq-length", type=int, default=30)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--stable-count", type=int, default=3)
    parser.add_argument("--no-save", action="store_true", help="Disable saving video")
    return parser.parse_args()

def main():
    args = parse_args()

    config = GestureConfig(
        camera_id=args.camera,
        model_path=args.model,
        actions=args.actions,
        seq_length=args.seq_length,
        threshold=args.threshold,
        stable_count=args.stable_count,
        save_video=not args.no_save
    )

    pipeline = GesturePipeline(config)

    try:
        cap = open_camera(config.camera_id)
    except Exception as e:
        logger.error(f"Failed to open camera: {e}")
        return

    # Video writing logic
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = None
    if config.save_video:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        # fourcc code needs to be int or string? cv2.VideoWriter_fourcc returns int.
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_path = config.output_dir / "output.mp4"
        logger.info(f"Saving video to {output_path}")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    try:
        logger.info("Starting loop. Press 'q' to exit.")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame, prediction = pipeline.process_frame(frame)

            if writer:
                writer.write(processed_frame)

            cv2.imshow("Gesture Recognition", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        pipeline.close()

if __name__ == "__main__":
    main()
