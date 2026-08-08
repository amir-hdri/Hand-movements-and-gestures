import cv2
import threading
import time
import asyncio
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import numpy as np

from .config import config
from .data_manager import DataManager
from .model_trainer import ModelTrainer
from .inference import InferenceEngine
from gesture_recognition.features import hand_landmarks_to_feature_vector

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State
class AppState:
    def __init__(self):
        self.mode = "idle"  # idle, recording, training, predicting
        self.data_manager = DataManager()
        self.model_trainer = ModelTrainer()
        self.inference_engine = InferenceEngine()
        self.training_status = "idle"
        self.last_prediction = {"action": None, "confidence": 0.0}
        self.latest_frame = None
        self.prediction_history = []
        self.history_lock = threading.Lock()
        self.lock = threading.Lock()
        # Tracks the most recently recorded history action so we do not flood
        # the history with one entry per camera frame for the same gesture.
        self.last_history_action = None
        self.last_history_confidence = 0.0

        # Load model initially
        self.reload_model()

    def add_prediction_to_history(self, action, confidence):
        """Add a prediction to the history, keeping only the last 100 entries.

        Consecutive identical predictions are collapsed into a single entry so
        the history does not get spammed at the camera frame rate.
        """
        if action is None:
            return
        with self.history_lock:
            # Skip duplicate consecutive entries for the same action.
            if self.last_history_action == action:
                return
            self.last_history_action = action
            self.last_history_confidence = float(confidence)
            self.prediction_history.append({
                "action": action,
                "confidence": float(confidence),
                "timestamp": datetime.now().isoformat()
            })
            # Keep only last 100 predictions
            if len(self.prediction_history) > 100:
                self.prediction_history = self.prediction_history[-100:]

    def get_prediction_history(self):
        """Get a copy of the prediction history"""
        with self.history_lock:
            return list(self.prediction_history)

    def clear_prediction_history(self):
        """Clear the prediction history"""
        with self.history_lock:
            self.prediction_history = []
            self.last_history_action = None
            self.last_history_confidence = 0.0

    def reload_model(self):
        actions = self.data_manager.get_available_gestures()
        success = self.inference_engine.load_model(actions)
        if success:
            print(f"Model loaded with actions: {actions}")
        else:
            print("Model failed to load or not found.")

state = AppState()

# Drawing helper: MediaPipe `solutions` provides drawing_utils + HAND_CONNECTIONS.
# It is optional — if unavailable, landmark overlay drawing is simply skipped.
try:
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    if mp_drawing is None or mp_hands is None:
        raise AttributeError
except (ImportError, AttributeError):
    print("Warning: MediaPipe solutions not found. Landmark drawing will be disabled.")
    mp_drawing = None
    mp_hands = None


def draw_landmarks_on_image(rgb_image, detection_result):
    if not mp_drawing or not mp_hands:
        return np.copy(rgb_image)

    hand_landmarks_list = detection_result.hand_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])

        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=hand_landmarks_proto,
            connections=mp_hands.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2))
    return annotated_image

# Camera Thread
class CameraThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.running = True
        self.cap = cv2.VideoCapture(0)
        self.detector = None

        # Initialize HandLandmarker (only if the model asset is available, so a
        # missing download does not crash the whole backend at import time).
        model_asset = config.MODELS_DIR / "hand_landmarker.task"
        if not model_asset.exists():
            print(f"Warning: HandLandmarker model not found at {model_asset}. "
                  "Camera detection will be disabled until the model is available.")
        else:
            try:
                base_options = python.BaseOptions(model_asset_path=str(model_asset))
                options = vision.HandLandmarkerOptions(
                    base_options=base_options,
                    num_hands=1,
                    min_hand_detection_confidence=0.5,
                    min_hand_presence_confidence=0.5,
                    min_tracking_confidence=0.5)
                self.detector = vision.HandLandmarker.create_from_options(options)
            except Exception as e:  # pragma: no cover - hardware/GL dependent
                print(f"Warning: Failed to initialise HandLandmarker: {e}. "
                      "Camera detection will be disabled.")

    def run(self):
        while self.running:
            if not self.cap.isOpened():
                time.sleep(1)
                self.cap.open(0)
                continue

            success, frame = self.cap.read()
            if not success:
                time.sleep(0.1)
                continue

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if self.detector is None:
                # No detector available: still stream the raw frame so the UI
                # shows something, and do not spin the CPU on detection.
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    with state.lock:
                        state.latest_frame = buffer.tobytes()
                time.sleep(0.05)
                continue

            # Create MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Detect
            detection_result = self.detector.detect(mp_image)

            # Draw
            annotated_frame = draw_landmarks_on_image(frame, detection_result)

            # Logic
            if detection_result.hand_landmarks:
                if state.mode == "recording":
                    state.data_manager.process_frame(detection_result.hand_landmarks)
                    cv2.putText(annotated_frame, "RECORDING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                if state.mode != "training" and state.inference_engine.model:
                    # Assuming single hand
                    hand_landmarks = detection_result.hand_landmarks[0]
                    fv = hand_landmarks_to_feature_vector(hand_landmarks)
                    pred = state.inference_engine.predict(fv)

                    if pred and pred.confidence > 0:
                        label_text = f"{pred.raw_action} ({pred.confidence:.2f})"
                        color = (0, 255, 0)
                        if pred.stable_action:
                            label_text = f"STABLE: {pred.stable_action}"
                            color = (255, 0, 0)
                            state.last_prediction = {"action": pred.stable_action, "confidence": pred.confidence}
                            # Only add stable actions to history; duplicates are
                            # collapsed inside add_prediction_to_history.
                            state.add_prediction_to_history(pred.stable_action, pred.confidence)
                        else:
                            state.last_prediction = {"action": pred.raw_action, "confidence": pred.confidence}

                        cv2.putText(annotated_frame, label_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    else:
                        state.last_prediction = {"action": None, "confidence": 0.0}

            # Update global frame
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            if ret:
                with state.lock:
                    state.latest_frame = buffer.tobytes()

            time.sleep(0.01)

    def stop(self):
        self.running = False
        self.cap.release()

camera_thread = CameraThread()
camera_thread.daemon = True
camera_thread.start()

# Video Stream Generator
async def generate_frames():
    while True:
        with state.lock:
            frame = state.latest_frame

        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            await asyncio.sleep(0.1)

        await asyncio.sleep(0.03)

# Pydantic Models
class RecordRequest(BaseModel):
    label: str

class ConfigUpdateRequest(BaseModel):
    seq_length: Optional[int] = None
    threshold: Optional[float] = None
    stable_count: Optional[int] = None
    smart_thresholds: Optional[Dict[str, float]] = None

class AddGestureRequest(BaseModel):
    label: str

# Endpoints

@app.get("/api/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/status")
async def get_status():
    return {
        "mode": state.mode,
        "training_status": state.training_status,
        "last_prediction": state.last_prediction
    }

@app.post("/api/record/start")
async def start_recording(req: RecordRequest):
    if state.mode == "training":
        raise HTTPException(status_code=400, detail="System is training")

    try:
        state.data_manager.start_recording(req.label)
        state.mode = "recording"
        return {"status": "started", "label": req.label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/record/stop")
async def stop_recording():
    if state.mode != "recording":
        raise HTTPException(status_code=400, detail="Not recording")

    try:
        state.data_manager.stop_recording()
        state.mode = "idle"
        return {"status": "stopped"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def training_task():
    state.mode = "training"
    state.training_status = "training"
    try:
        actions = state.data_manager.get_available_gestures()
        print(f"Starting training for actions: {actions}")
        state.model_trainer.train(actions)
        state.training_status = "completed"
        state.reload_model()
    except Exception as e:
        print(f"Training failed: {e}")
        state.training_status = "failed"
    finally:
        state.mode = "idle"

@app.post("/api/train")
async def train_model(background_tasks: BackgroundTasks):
    if state.mode != "idle":
        raise HTTPException(status_code=400, detail=f"System is busy ({state.mode})")

    background_tasks.add_task(training_task)
    return {"status": "training_started"}

@app.get("/api/gestures")
async def get_gestures():
    return {"gestures": state.data_manager.get_available_gestures()}

@app.post("/api/gestures")
async def add_gesture(req: AddGestureRequest):
    try:
        state.data_manager.add_gesture(req.label)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "added", "label": req.label}

@app.delete("/api/gestures")
async def delete_gesture(req: AddGestureRequest):
    try:
        state.data_manager.remove_gesture(req.label)
        # Reload model with updated gestures
        state.reload_model()
        return {"status": "deleted", "label": req.label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Prediction History Endpoints
@app.get("/api/history")
async def get_history():
    return {"history": state.get_prediction_history()}

@app.delete("/api/history")
async def clear_history():
    state.clear_prediction_history()
    return {"status": "cleared"}

# Dataset Management Endpoints
@app.post("/api/dataset/export")
async def export_dataset():
    try:
        export_path = state.data_manager.export_dataset()
        return {"status": "exported", "path": str(export_path)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/dataset/reset")
async def reset_dataset():
    try:
        state.data_manager.reset_dataset()
        state.reload_model()
        return {"status": "reset"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/config")
async def get_config():
    # Report the sequence length actually in use (derived from the loaded model
    # when available) so the UI reflects the real value, not just config.
    seq_length = config.SEQ_LENGTH
    recognizer = getattr(state.inference_engine, "recognizer", None)
    if recognizer is not None:
        seq_length = recognizer.seq_length
    return {
        "seq_length": seq_length,
        "threshold": config.DEFAULT_THRESHOLD,
        "stable_count": config.STABLE_COUNT,
        "smart_thresholds": config.SMART_THRESHOLDS,
    }

@app.post("/api/config")
async def update_config(req: ConfigUpdateRequest):
    if req.seq_length is not None:
        if req.seq_length <= 0:
            raise HTTPException(status_code=400, detail="seq_length must be > 0")
        config.SEQ_LENGTH = int(req.seq_length)
    if req.threshold is not None:
        if not 0.0 < req.threshold <= 1.0:
            raise HTTPException(status_code=400, detail="threshold must be in (0, 1]")
        config.DEFAULT_THRESHOLD = float(req.threshold)
    if req.stable_count is not None:
        if req.stable_count <= 0:
            raise HTTPException(status_code=400, detail="stable_count must be > 0")
        config.STABLE_COUNT = int(req.stable_count)
    if req.smart_thresholds is not None:
        config.SMART_THRESHOLDS.update(req.smart_thresholds)

    # Rebuild the recognizer so the new sequence length / threshold / stable
    # count take effect, then push updated smart thresholds.
    state.reload_model()
    state.inference_engine.update_thresholds(config.SMART_THRESHOLDS)

    return {
        "status": "updated",
        "seq_length": config.SEQ_LENGTH,
        "threshold": config.DEFAULT_THRESHOLD,
        "stable_count": config.STABLE_COUNT,
        "smart_thresholds": config.SMART_THRESHOLDS,
    }

# Serve Frontend
frontend_path = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="static")
else:
    print(f"Frontend build not found at {frontend_path}. Running API only.")
