from __future__ import annotations
import os
from pathlib import Path

import numpy as np


def main() -> int:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    project_root = Path(__file__).resolve().parent.parent
    model_path = project_root / "models" / "model2_1.0.h5"

    import tensorflow as tf
    import keras

    if not model_path.exists():
        raise SystemExit(f"Model file not found: {model_path}")

    # Basic environment sanity: this project currently expects the TF/Keras 2.x ecosystem.
    np_version = np.__version__
    tf_version = tf.__version__
    keras_version = getattr(keras, "__version__", "unknown")
    print(f"numpy={np_version} tensorflow={tf_version} keras={keras_version}")

    try:
        if int(np_version.split(".", 1)[0]) >= 2:
            raise SystemExit(
                "NumPy 2.x detected. This project currently requires NumPy 1.x.\n"
                "Fix: pip install --force-reinstall -r requirements.txt"
            )
        if keras_version != "unknown" and int(keras_version.split(".", 1)[0]) >= 3:
            raise SystemExit(
                "Keras 3.x detected. The shipped .h5 model was trained/saved with Keras 2.x.\n"
                "Fix: pip install --force-reinstall -r requirements.txt"
            )
    except ValueError:
        # If version parsing fails, keep going and let import/model-load errors surface.
        pass

    model = tf.keras.models.load_model(model_path, compile=False)
    dummy = np.zeros((1, 30, 99), dtype=np.float32)
    _ = model.predict(dummy, verbose=0)

    import cv2  # noqa: F401
    import mediapipe as mp  # noqa: F401
    if not hasattr(mp, "solutions"):
        raise SystemExit(
            "MediaPipe 'solutions' module is missing. Install mediapipe==0.10.14\n"
            "and ensure dependencies are installed via requirements.txt."
        )

    print("smoke ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
