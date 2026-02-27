from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from gesture_recognition.features import append_label, hand_landmarks_to_feature_vector


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Collect hand gesture dataset from a webcam.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument(
        "--actions",
        nargs="+",
        default=["come", "away", "spin"],
        help="Action names (order becomes label indices)",
    )
    parser.add_argument("--seq-length", type=int, default=30, help="Sequence length (default: 30)")
    parser.add_argument(
        "--secs-for-action", type=int, default=30, help="Seconds to record per action (default: 30)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "dataset",
        help="Directory to write .npy files (default: ./dataset)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import cv2
        import mediapipe as mp
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise SystemExit(
            "Missing dependencies. Install: opencv-python, mediapipe, numpy"
        ) from exc

    created_time = int(time.time())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.seq_length <= 0:
        raise SystemExit("--seq-length must be > 0")
    if args.secs_for_action <= 0:
        raise SystemExit("--secs-for-action must be > 0")

    # MediaPipe hands model
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}")

    should_stop = False
    try:
        for label_index, action in enumerate(args.actions):
            data: list[np.ndarray] = []

            ret, img = cap.read()
            if not ret:
                print("Failed to read from camera.")
                break

            img = cv2.flip(img, 1)
            cv2.putText(
                img,
                f"Waiting for collecting {action.upper()} action...",
                org=(10, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(255, 255, 255),
                thickness=2,
            )
            cv2.imshow("img", img)
            cv2.waitKey(3000)

            start_time = time.time()
            while time.time() - start_time < args.secs_for_action:
                ret, img = cap.read()
                if not ret:
                    print("Failed to read from camera.")
                    should_stop = True
                    break

                img = cv2.flip(img, 1)
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)

                if result.multi_hand_landmarks is not None:
                    for res in result.multi_hand_landmarks:
                        fv = hand_landmarks_to_feature_vector(res)
                        data.append(append_label(fv, label_index))
                        mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

                cv2.imshow("img", img)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    should_stop = True
                    break

            data_arr = np.asarray(data, dtype=np.float32)
            print(action, data_arr.shape)
            np.save(args.output_dir / f"raw_{action}_{created_time}.npy", data_arr)

            # Create sequence data (includes label in the last column).
            if len(data_arr) >= args.seq_length:
                full_seq_arr = (
                    np.lib.stride_tricks.sliding_window_view(
                        data_arr, (args.seq_length, data_arr.shape[1])
                    )
                    .squeeze(axis=1)
                    .astype(np.float32)
                )
            else:
                full_seq_arr = np.asarray([], dtype=np.float32)
            print(action, full_seq_arr.shape)
            np.save(args.output_dir / f"seq_{action}_{created_time}.npy", full_seq_arr)

            if should_stop:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
