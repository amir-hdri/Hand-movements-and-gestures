#!/usr/bin/env python3
"""Convert the legacy Keras-2 H5 model (model2_1.0.h5) into a Keras-3-compatible
model so it can be loaded on modern TensorFlow/Keras environments.

The legacy checkpoint was saved with Keras 2.4 and stores an LSTM config
containing the ``time_major`` kwarg, which Keras 3 rejects.  This script
re-builds the identical architecture (LSTM(64) -> Dense(32) -> Dense(3)) and
copies the trained weights over, then writes a fresh Keras-3-compatible model.

Usage:
    python scripts/convert_legacy_model.py [src] [dst]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else project_root / "models" / "model2_1.0.h5"
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src

    import tensorflow as tf

    try:
        # Try loading directly first (e.g. running under Keras 2 or a newer Keras
        # that tolerates the old config).  If it works there is nothing to do.
        model = tf.keras.models.load_model(src, compile=False)
        print(f"Legacy model {src} loads directly; no conversion needed.")
    except Exception:
        # Extract the architecture from the legacy config.
        import h5py

        with h5py.File(src, "r") as f:
            legacy_weights = {}
            for name, obj in f["model_weights"].items():
                def _collect(group, prefix=""):
                    for key, value in group.items():
                        path = f"{prefix}/{key}" if prefix else key
                        if hasattr(value, "items"):
                            _collect(value, path)
                        else:
                            legacy_weights[path] = np.array(value[()])
                _collect(obj)

        seq_length, n_features = 30, 99
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(seq_length, n_features)),
                tf.keras.layers.LSTM(64),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(3, activation="softmax"),
            ],
            name="sequential",
        )

        # Match layer order: LSTM, Dense(32), Dense(3).
        lstm = model.layers[0]
        lstm.set_weights(
            [
                legacy_weights["lstm/lstm_cell/kernel:0"],
                legacy_weights["lstm/lstm_cell/recurrent_kernel:0"],
                legacy_weights["lstm/lstm_cell/bias:0"],
            ]
        )
        dense1 = model.layers[1]
        dense1.set_weights(
            [
                legacy_weights["dense/kernel:0"],
                legacy_weights["dense/bias:0"],
            ]
        )
        dense2 = model.layers[2]
        dense2.set_weights(
            [
                legacy_weights["dense_1/kernel:0"],
                legacy_weights["dense_1/bias:0"],
            ]
        )

        print(f"Rebuilt and converted legacy model from {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    model.save(dst)
    print(f"Saved converted model to {dst}")

    # Sanity check: load it back and run a prediction.
    reloaded = tf.keras.models.load_model(dst, compile=False)
    dummy = np.zeros((1, seq_length, n_features), dtype=np.float32)
    out = reloaded(dummy, training=False).numpy()
    print(f"Reload OK: input={reloaded.input_shape} output={reloaded.output_shape} "
          f"sample_out={np.round(out, 4).tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
