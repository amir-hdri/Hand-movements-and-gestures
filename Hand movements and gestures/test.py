from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from gesture_recognition.features import hand_landmarks_to_feature_vector
from gesture_recognition.recognizer import GestureRecognizer


# Default to 2 actions to match the shipped model.h5
DEFAULT_ACTIONS = ["come", "away"]


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Realtime gesture recognition demo (webcam/video).")
    parser.add_argument(
        "--model",
        type=Path,
        default=project_root / "models" / "model.h5",
        help="Path to a Keras .h5 model (default: ./models/model.h5)",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index (e.g. 0) or a video file path (default: 0)",
    )
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
        help="Require N consecutive predictions to show action (default: 3)",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=project_root / "artifacts" / "videos",
        help="Directory to save input/output mp4 (default: ./artifacts/videos)",
    )
    parser.add_argument("--no-save", action="store_true", help="Disable saving video to disk")
    return parser.parse_args()


def _open_capture(source: str):
    import cv2

    if source.isdigit():
        return cv2.VideoCapture(int(source))
    return cv2.VideoCapture(source)


def main() -> int:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    args = parse_args()

    try:
        import cv2
        import mediapipe as mp
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise SystemExit(
            "Missing dependencies. Install: opencv-python, mediapipe, tensorflow, numpy"
        ) from exc

    if args.seq_length <= 0:
        raise SystemExit("--seq-length must be > 0")
    if args.stable_count <= 0:
        raise SystemExit("--stable-count must be > 0")

    if not args.model.exists():
        # Fallback to model2_1.0.h5 if model.h5 doesn't exist (though strictly model.h5 is preferred)
        legacy_model = args.model.parent / "model2_1.0.h5"
        if args.model.name == "model.h5" and legacy_model.exists():
             print(f"Warning: model.h5 not found, falling back to {legacy_model}")
             args.model = legacy_model
             # If falling back, we might need to adjust actions if they differ, but we can't know for sure.
        else:
            raise SystemExit(f"Model file not found: {args.model}")

    # Load model
    try:
        model = tf.keras.models.load_model(args.model, compile=False)
    except Exception as e:
        raise SystemExit(f"Failed to load model: {e}")

    recognizer = GestureRecognizer(
        model,
        args.actions,
        seq_length=args.seq_length,
        threshold=args.threshold,
        stable_count=args.stable_count,
    )

    # MediaPipe hands model
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = _open_capture(str(args.source))
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    # Prime one frame to determine resolution for writers.
    ret, frame = cap.read()
    if not ret:
        raise SystemExit("Could not read from source.")

    h, w = frame.shape[:2]
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps < 1.0 or np.isnan(fps):
        fps = 30.0

    writer_in = None
    writer_out = None
    if not args.no_save:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
        writer_in = cv2.VideoWriter(str(args.save_dir / "input.mp4"), fourcc, fps, (w, h))
        writer_out = cv2.VideoWriter(str(args.save_dir / "output.mp4"), fourcc, fps, (w, h))

    last_overlay = "?"
    last_overlay_time = time.time()
    try:
        while True:
            img0 = frame.copy()

            img = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks is not None:
                for res in result.multi_hand_landmarks:
                    fv = hand_landmarks_to_feature_vector(res)
                    pred = recognizer.update(fv)

                    mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

                    if pred.stable_action is not None:
                        last_overlay = pred.stable_action.upper()
                        last_overlay_time = time.time()

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

            # Fade back to '?' if we haven't seen a stable action recently.
            if time.time() - last_overlay_time > 1.5:
                last_overlay = "?"

            if writer_in is not None:
                writer_in.write(img0)
            if writer_out is not None:
                writer_out.write(img)

            cv2.imshow("img", img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            ret, frame = cap.read()
            if not ret:
                break
    finally:
        cap.release()
        if writer_in is not None:
            writer_in.release()
        if writer_out is not None:
            writer_out.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
