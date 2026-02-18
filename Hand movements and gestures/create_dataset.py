import argparse
import time
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# Ensure project root is in path
sys.path.append(str(Path(__file__).resolve().parent))

from gesture_recognition.config import GestureConfig
from gesture_recognition.features import append_label, hand_landmarks_to_feature_vector
from gesture_recognition.utils import setup_logging, open_camera

logger = setup_logging(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect hand gesture dataset from a webcam.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    parser.add_argument("--actions", nargs="+", default=["come", "away", "spin"])
    parser.add_argument("--seq-length", type=int, default=30)
    parser.add_argument("--secs-for-action", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=Path("dataset"))
    return parser.parse_args()

def main():
    args = parse_args()

    config = GestureConfig(
        camera_id=args.camera,
        actions=args.actions,
        seq_length=args.seq_length,
        secs_for_action=args.secs_for_action,
        dataset_output_dir=args.output_dir
    )

    config.dataset_output_dir.mkdir(parents=True, exist_ok=True)
    created_time = int(time.time())

    # MediaPipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    try:
        cap = open_camera(config.camera_id)
    except Exception as e:
        logger.error(f"Failed to open camera: {e}")
        return

    try:
        logger.info("Starting dataset collection. Press 'q' to abort.")

        for label_index, action in enumerate(config.actions):
            data = []

            logger.info(f"Preparing to collect action: {action}")

            # Countdown
            start_wait = time.time()
            while time.time() - start_wait < 3.0:
                 ret, img = cap.read()
                 if not ret:
                     break

                 img = cv2.flip(img, 1)
                 cv2.putText(
                    img,
                    f"Collecting {action.upper()} in {3.0 - (time.time() - start_wait):.1f}s",
                    org=(10, 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(255, 255, 255),
                    thickness=2,
                )
                 cv2.imshow("Dataset Collection", img)
                 if cv2.waitKey(1) & 0xFF == ord('q'):
                     return

            start_time = time.time()
            logger.info(f"Collecting {action}...")

            while time.time() - start_time < config.secs_for_action:
                ret, img = cap.read()
                if not ret:
                    break

                img = cv2.flip(img, 1)
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)

                if result.multi_hand_landmarks:
                    for res in result.multi_hand_landmarks:
                        fv = hand_landmarks_to_feature_vector(res)
                        data.append(append_label(fv, label_index))
                        mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

                cv2.imshow("Dataset Collection", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return

            data_arr = np.asarray(data, dtype=np.float32)
            logger.info(f"Collected {action}: {data_arr.shape}")

            if len(data_arr) == 0:
                logger.warning(f"No data collected for {action}!")
                continue

            np.save(config.dataset_output_dir / f"raw_{action}_{created_time}.npy", data_arr)

            # Create sequence data
            full_seq_data = []
            if len(data_arr) >= config.seq_length:
                for start in range(0, len(data_arr) - config.seq_length + 1):
                    full_seq_data.append(data_arr[start : start + config.seq_length])

                full_seq_arr = np.asarray(full_seq_data, dtype=np.float32)
                logger.info(f"Sequence data {action}: {full_seq_arr.shape}")
                np.save(config.dataset_output_dir / f"seq_{action}_{created_time}.npy", full_seq_arr)
            else:
                logger.warning(f"Not enough data for sequence length {config.seq_length}")

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()

if __name__ == "__main__":
    main()
