from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import asyncio
import logging
import sys
from pathlib import Path
from pydantic import BaseModel
from typing import List, Optional

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from gesture_recognition.gui.model_trainer import ModelTrainer
from gesture_recognition.gui.inference import InferenceEngine

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Gesture Recognition Dashboard")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_trainer = ModelTrainer()
inference_engine = InferenceEngine()

# Pydantic models
class RecordRequest(BaseModel):
    action: str = ""
    start: bool
    label_index: int = 0

class TrainRequest(BaseModel):
    actions: List[str]
    epochs: int = 50

class InferenceRequest(BaseModel):
    model: str
    robot: bool = False

class SettingsRequest(BaseModel):
    camera: int

@app.get("/api/status")
async def get_status():
    status = "Idle"
    if inference_engine.dataset_manager.recording:
        status = "Recording"
    elif inference_engine.is_running and inference_engine.recognizer:
        status = "Inference Running"
    elif model_trainer.is_training:
        status = "Training"
    elif inference_engine.is_running:
        status = "Camera On"
    return {"status": status}

@app.get("/api/dataset/actions")
async def get_actions():
    actions = model_trainer.get_available_actions()
    return {"actions": actions}

@app.post("/api/dataset/record")
async def record_dataset(req: RecordRequest):
    if req.start:
        if not inference_engine.is_running:
            inference_engine.start_camera()

        inference_engine.dataset_manager.start_recording(
            req.action, req.label_index
        )
        return {"status": "started", "action": req.action}
    else:
        inference_engine.dataset_manager.stop_recording()
        return {"status": "stopped"}

@app.post("/api/train")
async def train_model(req: TrainRequest, background_tasks: BackgroundTasks):
    if model_trainer.is_training:
        return {"status": "error", "message": "Training already in progress"}

    def train_task():
        result = model_trainer.train_model(req.actions, req.epochs)
        logger.info(f"Training finished: {result}")

    background_tasks.add_task(train_task)
    return {"status": "started", "message": "Training started in background"}

@app.get("/api/models")
async def get_models():
    models = []
    if model_trainer.models_dir.exists():
        for f in model_trainer.models_dir.glob("*"):
            if f.suffix in [".h5", ".keras"]:
                models.append(f.name)
    return {"models": sorted(models)}

@app.post("/api/inference/start")
async def start_inference(req: InferenceRequest):
    model_path = model_trainer.models_dir / req.model
    if not model_path.exists():
        return {"status": "error", "message": "Model not found"}

    import json
    actions_path = model_trainer.models_dir / f"{req.model}_labels.json"
    actions = []
    if actions_path.exists():
        with open(actions_path, "r") as f:
            actions = json.load(f)
    else:
        actions = model_trainer.get_available_actions()

    inference_engine.load_model(model_path, actions)

    if req.robot:
        inference_engine.set_robot_controller("pingpong")
    else:
        inference_engine.set_robot_controller("mock")

    if not inference_engine.is_running:
        inference_engine.start_camera()

    return {"status": "started"}

@app.post("/api/inference/stop")
async def stop_inference():
    inference_engine.stop_camera()
    inference_engine.recognizer = None
    return {"status": "stopped"}

@app.post("/api/settings")
async def save_settings(req: SettingsRequest):
    if inference_engine.camera_index != req.camera:
        inference_engine.stop_camera()
        inference_engine.start_camera(req.camera)
    return {"status": "saved"}

# Video Streaming
async def gen_frames():
    while True:
        frame_bytes = await inference_engine.get_frame()
        if frame_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            await asyncio.sleep(0.1)

@app.get("/api/video_feed")
async def video_feed():
    if not inference_engine.is_running:
         inference_engine.start_camera()

    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

# Mount static files
frontend_dir = Path(__file__).resolve().parent / "frontend" / "dist"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
