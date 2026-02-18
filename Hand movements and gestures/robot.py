from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

from gesture_recognition.features import hand_landmarks_to_feature_vector
from gesture_recognition.recognizer import GestureRecognizer


DEFAULT_ACTIONS = ["come", "away", "spin"]


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Gesture control loop for PingPong Robot (webcam).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=project_root / "models" / "model2_1.0.h5",
        help="Path to a Keras .h5 model (default: ./models/model2_1.0.h5)",
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument(
        "--actions",
        nargs="+",
        default=DEFAULT_ACTIONS,
        help="Action names in model output order",
    )
    parser.add_argument("--seq-length", type=int, default=30, help="Sequence length (default: 30)")
    parser.add_argument("--threshold", type=float, default=0.9, help="Min confidence (default: 0.9)")
    parser.add_argument(
        "--stable-count",
        type=int,
        default=3,
        help="Require N consecutive predictions to trigger (default: 3)",
    )
    parser.add_argument(
        "--enable-robot",
        action="store_true",
        help="Enable PingPong robot control (requires hardware connected)",
    )
    return parser.parse_args()


def _ensure_pingpong_on_path() -> None:
    # Backwards compatible: keep the vendored pingpong dir importable even when
    # users run this script from a different working directory.
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def main() -> int:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    args = parse_args()

    try:
        import cv2
        import mediapipe as mp
        import tensorflow as tf
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise SystemExit(
            "Missing dependencies. Install: opencv-python, mediapipe, tensorflow, numpy"
        ) from exc

    if not args.model.exists():
        raise SystemExit(f"Model file not found: {args.model}")

    model = tf.keras.models.load_model(args.model, compile=False)
    recognizer = GestureRecognizer(
        model,
        args.actions,
        seq_length=args.seq_length,
        threshold=args.threshold,
        stable_count=args.stable_count,
    )

    # Optional: PingPong robot integration (kept minimal and off by default).
    pingpong = None
    if args.enable_robot:
        _ensure_pingpong_on_path()
        try:
            from pingpong import PingPongThread  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise SystemExit(
                "PingPong library not available/importable. "
                "Check ./pingpong and required dependencies (e.g. pyserial)."
            ) from exc

        pingpong = PingPongThread(number=2)
        pingpong.start()
        pingpong.wait_until_full_connect()

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

    last_action: str | None = None
    last_overlay = "?"
    last_overlay_time = time.time()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            img = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks is not None:
                for res in result.multi_hand_landmarks:
                    fv = hand_landmarks_to_feature_vector(res)
                    pred = recognizer.update(fv)

                    mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

                    if pred.stable_action is not None:
                        this_action = pred.stable_action
                        last_overlay = this_action.upper()
                        last_overlay_time = time.time()

                        if pingpong is not None and last_action != this_action:
                            # Map gestures to robot actions here.
                            # Example skeleton (commented):
                            # if this_action == "come": ...
                            # elif this_action == "away": ...
                            # elif this_action == "spin": ...
                            last_action = this_action

                    anchor = res.landmark[0]
                    cv2.putText(
                        img,
                        last_overlay,
                        org=(int(anchor.x * img.shape[1]), int(anchor.y * img.shape[0] + 20)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=(255, 255, 255),
                        thickness=2,
                    )

            if time.time() - last_overlay_time > 1.5:
                last_overlay = "?"

            cv2.imshow("img", img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if pingpong is not None:
            try:
                pingpong.end()
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
