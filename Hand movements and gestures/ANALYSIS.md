# Project Analysis & Fix Report

This document records a full audit of the Hand Gesture Recognition project, the
bugs/gaps found, and the fixes applied. All fixes are verified by the test suite
(`51 passed`) and by loading the shipped models under modern TensorFlow/Keras.

---

## Environment verified

- Python 3.11, NumPy 1.26.4, TensorFlow 2.16.2, Keras 3.15, MediaPipe 0.10.9, OpenCV 4.9.
- Both shipped models load and predict correctly:
  - `models/model.h5` — 2 classes (`come`, `away`), input `(None, 30, 99)`.
  - `models/model2_1.0.h5` — 3 classes (`come`, `away`, `spin`), input `(None, 30, 99)`.
- Frontend builds cleanly with `npm run build` (Vite).

---

## Critical bugs found & fixed

### 1. `model2_1.0.h5` could not be loaded under Keras 3
It was saved with Keras 2.4 and stored an LSTM config with a `time_major` kwarg
that Keras 3 rejects. `robot.py` defaults to this model, so it crashed on startup
on any modern environment.
**Fix:** Added `scripts/convert_legacy_model.py` which rebuilds the identical
architecture and copies the trained weights, then re-saves a Keras-3-compatible
model. Verified it produces **bit-identical predictions** to the original weights.
`test.py` now also prints a clear message pointing to the converter.

### 2. Frontend would not build (`npm run build` failed)
Two hard build failures:
- `GestureHistory.jsx` imported `date-fns` which was **not in `package.json`**.
- `CameraView.jsx` imported `RecordCircle`, which is **not a valid MUI icon** export.
**Fix:** replaced `date-fns` with native `Date` formatting, and `RecordCircle` with
`FiberManualRecord`.

### 3. Prediction history was spammed one entry per camera frame
In `app.py` the camera thread called `add_prediction_to_history()` on **every**
frame that produced a prediction (dozens of duplicate rows per second). The React
`App.jsx` duplicated this with a per-second poll effect that added a new entry each
time because `last_prediction` is a fresh object every poll.
**Fix:** `AppState` now collapses consecutive identical entries; `App.jsx` tracks the
last seen action string. Only *stable* actions are recorded to history.

### 4. "Stable" detection was not truly consecutive
`GestureRecognizer` kept a deque of recent above-threshold actions and did not
clear it when a below-threshold frame arrived, so two separated detections of the
same action could count as consecutive.
**Fix:** a below-threshold frame now resets the run (both in `GestureRecognizer` and
`SmartGestureRecognizer`).

### 5. `SmartGestureRecognizer` used the slow `.predict()` path
The subclass overrode the fast `tf.function`/direct-call path used by the base
recognizer and called `model.predict()` per frame — a large inference performance
regression.
**Fix:** it now uses the direct `self._model(input_data, training=False)` path.

### 6. Settings in the UI did nothing
`SettingsPanel`'s "Save Settings" only updated local React state; nothing was sent
to the backend, and the backend `/api/config` expected a different schema.
**Fix:** `SettingsPanel` now calls the backend, and the `/api/config` POST endpoint
accepts/applies `seq_length`, `threshold`, `stable_count`, and `smart_thresholds`
(rebuilding the recognizer so changes take effect immediately).

### 7. Sequence-length mismatch could crash inference
If `SEQ_LENGTH` changed in settings without retraining, the recognizer would feed
windows of the wrong length to the model.
**Fix:** `InferenceEngine.load_model` now derives the sequence length from the loaded
model's input shape (falling back to config), and warns if the model's class count
doesn't match the configured gestures.

### 8. `remove_gesture` used substring matching
Deleting `spin` also deleted `spin2`/`left_spin` data files.
**Fix:** match only files with the exact `raw_<label>_` / `seq_<label>_` prefixes.

### 9. Unused mediapipe imports made the data manager heavy to import
`data_manager.py` imported mediapipe at module scope although none of its methods
use it, forcing a heavy dependency import (and breaking unit tests that mocked it).
**Fix:** removed the unused imports. The module is now importable without mediapipe.

### 10. Dead / broken code in the model trainer
`ModelTrainer.load_data()` was a stub containing only `pass`.
**Fix:** removed it. Training now also refuses to run with fewer than 2 gestures
with data (a 1-class softmax is not a meaningful classifier).

### 11. Graceful failure when the MediaPipe model is missing
`app.py` created the `HandLandmarker` unconditionally at import; a missing
`hand_landmarker.task` (or a GL/GPU failure) would crash the whole backend. The
camera thread also had a broken indentation around the recording block.
**Fix:** detector creation is guarded and fails soft (raw frames still stream);
indentation cleaned; `process_frame()` signature cleaned of an unused argument.

### 12. Dependency conflict in `requirements.txt`
`numpy<2.0` conflicts with `mediapipe>=0.10.14` (pulls `jax`, which needs numpy>=2)
and with `opencv-contrib-python>=5`.
**Fix:** pinned `mediapipe>=0.10.9,<0.10.14` and `opencv-contrib-python<5`, and added
`python-dotenv` for `.env` support.

---

## Gaps addressed

- **`.env` support** — `config.py` now loads `.env` (README documents it) and reads
  `SEQ_LENGTH`, `CONFIDENCE_THRESHOLD`, `STABLE_COUNT`, `SECS_FOR_ACTION`, and the
  existing `CORS_ALLOWED_ORIGINS` from the environment.
- **Gesture label validation** — empty labels and labels containing path separators
  are rejected (avoids bad filenames / path traversal).
- **Model/action validation in `test.py`** — a clear error is raised when the model
  output count doesn't match the `--actions` list, instead of a cryptic shape error.
- **`.gitignore`** — now excludes `.venv/`, `node_modules/`, `dist/`, `.env`,
  `artifacts/`, and export zips.
- **README** — documents the new config API, env vars, and Keras-3 model conversion.

---

## Test coverage added

| File | What it covers |
|------|----------------|
| `tests/test_recognizer.py` | Below-threshold frame resets a stable run. |
| `tests/test_inference_smart.py` | Fast inference path (no `.predict`), smart thresholds, low-confidence reset, shape-mismatch handling. |
| `tests/test_app_state.py` | History deduplication / capping / clearing. |
| `tests/test_config.py` | Env-var overrides and invalid-value fallback. |
| `tests/test_label_manager.py` | Label validation and exact-match gesture removal (near-miss labels preserved). |

**Result:** `python -m pytest tests/ -q` → **51 passed** (up from 33).

---

## Verification performed

- Both models load and predict under TensorFlow 2.16 / Keras 3.
- Converted model predictions match the original Keras-2 weights exactly (max diff 0.0).
- Backend imports and serves the frontend; `/api/config` GET/POST validated via
  FastAPI `TestClient` (including validation of invalid thresholds).
- Frontend production build succeeds.
- `scripts/smoke.py` passes.

## Notes / limitations

- The sandbox has no physical camera or GL/EGL, so live video detection cannot be
  exercised here; the code paths for detection are exercised via unit tests and the
  same MediaPipe APIs the `create_dataset.py`/`robot.py` scripts use.
- Legacy `pingpong/` and `legacy_gui/` modules are unchanged (marked legacy); their
  existing tests still pass.
