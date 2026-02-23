from __future__ import annotations
import os
from pathlib import Path

import numpy as np


def main() -> int:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    project_root = Path(__file__).resolve().parent.parent
    # Prioritize new model
    model_path = project_root / "models" / "model.h5"
    if not model_path.exists():
        model_path = project_root / "models" / "model2_1.0.h5"

    import tensorflow as tf
    import keras

    if not model_path.exists():
        raise SystemExit(f"Model file not found: {model_path}")

    # Basic environment sanity
    np_version = np.__version__
    tf_version = tf.__version__
    keras_version = getattr(keras, "__version__", "unknown")
    print(f"numpy={np_version} tensorflow={tf_version} keras={keras_version}")

    # Check for incompatible numpy (2.x) if using older environment, but allow it if it works.
    try:
        if int(np_version.split(".", 1)[0]) >= 2:
            print("Warning: NumPy 2.x detected. Project recommends NumPy 1.x but proceeding.")
    except ValueError:
        pass

    # Keras 3 check: Removed strict block as new model.h5 supports Keras 3.
    # If using old model, load_model might fail, which is handled below.

    print(f"Loading model: {model_path}")
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        print(f"Failed to load model: {e}")
        # If trying to load old model with Keras 3, suggest fix.
        if "model2_1.0.h5" in str(model_path) and int(keras_version.split(".", 1)[0]) >= 3:
             print("Note: Keras 3.x detected with legacy model. Consider using 'model.h5' or downgrading environment.")
        raise

    # Create dummy input based on model expected shape
    # Assuming standard (None, 30, 99) or similar.
    # Inspect model input shape if possible, or use default.
    input_shape = (1, 30, 99) # Default for legacy
    if hasattr(model, "input_shape"):
        shape = model.input_shape
        # Handle tuple of shapes or single shape
        if isinstance(shape, tuple) and len(shape) > 1:
             # (None, 30, 99)
             input_shape = (1,) + shape[1:]

    try:
        dummy = np.zeros(input_shape, dtype=np.float32)
        _ = model.predict(dummy, verbose=0)
    except Exception as e:
        print(f"Prediction failed with shape {input_shape}: {e}")
        raise

    import cv2  # noqa: F401
    import mediapipe as mp  # noqa: F401
    # Check for solutions OR tasks API depending on version
    if not hasattr(mp, "solutions") and not hasattr(mp, "tasks"):
         raise SystemExit("MediaPipe installation seems incomplete.")

    print("smoke ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
