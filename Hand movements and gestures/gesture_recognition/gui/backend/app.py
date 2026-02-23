import cv2
import threading
import time
import asyncio
from typing import List, Dict, Optional
from pathlib import Path

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
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
    allow_origins=["*"],
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
        self.lock = threading.Lock()

        # Load model initially
        self.reload_model()

    def reload_model(self):
        actions = self.data_manager.get_available_gestures()
        success = self.inference_engine.load_model(actions)
        if success:
            print(f"Model loaded with actions: {actions}")
        else:
            print("Model failed to load or not found.")

state = AppState()

# Drawing Helper
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands # Still exists for constants like HAND_CONNECTIONS?
# Wait, if `solutions` is missing, I cannot use `mp.solutions.drawing_utils`.
# I should check if `mp.solutions` works now.
# But I am using tasks API for inference.
# `mp.solutions` might work for drawing utils if I install correct package?
# But if it failed before, it will fail here.
# I will implement custom drawing or try to import it safely.

try:
    from mediapipe.python.solutions import drawing_utils as mp_drawing
    from mediapipe.python.solutions import hands as mp_hands
except ImportError:
    try:
        mp_drawing = mp.solutions.drawing_utils
        mp_hands = mp.solutions.hands
    except ImportError:
        print("Warning: MediaPipe solutions not found. Drawing will be disabled.")
        mp_drawing = None
        mp_hands = None

def draw_landmarks_on_image(rgb_image, detection_result):
    if not mp_drawing:
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
            connections=mp_hands.HAND_CONNECTIONS if mp_hands else None,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2))
    return annotated_image

# Camera Thread
class CameraThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.running = True
        self.cap = cv2.VideoCapture(0)

        # Initialize HandLandmarker
        base_options = python.BaseOptions(model_asset_path=str(config.MODELS_DIR / "hand_landmarker.task"))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5)
        self.detector = vision.HandLandmarker.create_from_options(options)

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

            # Create MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Detect
            detection_result = self.detector.detect(mp_image)

            # Draw
            annotated_frame = draw_landmarks_on_image(frame, detection_result)

            # Logic
            if detection_result.hand_landmarks:
                if state.mode == "recording":
                        state.data_manager.process_frame(rgb_frame, detection_result.hand_landmarks)
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
def generate_frames():
    while True:
        with state.lock:
            frame = state.latest_frame

        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            time.sleep(0.1)

        time.sleep(0.03)

# Pydantic Models
class RecordRequest(BaseModel):
    label: str

class ThresholdConfig(BaseModel):
    thresholds: Dict[str, float]

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
    state.data_manager.add_gesture(req.label)
    return {"status": "added", "label": req.label}

@app.get("/api/config")
async def get_config():
    return {"smart_thresholds": config.SMART_THRESHOLDS}

@app.post("/api/config")
async def update_config(req: ThresholdConfig):
    config.SMART_THRESHOLDS.update(req.thresholds)
    state.inference_engine.update_thresholds(config.SMART_THRESHOLDS)
    return {"status": "updated", "smart_thresholds": config.SMART_THRESHOLDS}

# Serve Frontend
frontend_path = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="static")
else:
    print(f"Frontend build not found at {frontend_path}. Running API only.")
