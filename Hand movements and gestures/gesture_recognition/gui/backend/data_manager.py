import time
import json
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import threading
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from gesture_recognition.features import append_label, hand_landmarks_to_feature_vector
from .config import config

class LabelManager:
    def __init__(self, dataset_dir: Path):
        self.dataset_dir = dataset_dir
        self.labels_file = self.dataset_dir / "labels.json"
        self.labels: List[str] = self._load_labels()

    def _load_labels(self) -> List[str]:
        if self.labels_file.exists():
            with open(self.labels_file, "r") as f:
                return json.load(f)
        return config.ACTIONS

    def save_labels(self):
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        with open(self.labels_file, "w") as f:
            json.dump(self.labels, f)

    def get_labels(self) -> List[str]:
        return self.labels

    def add_label(self, label: str):
        if label not in self.labels:
            self.labels.append(label)
            self.save_labels()

    def get_label_index(self, label: str) -> int:
        if label not in self.labels:
            raise ValueError(f"Label {label} not found")
        return self.labels.index(label)

class DataManager:
    def __init__(self):
        self.label_manager = LabelManager(config.DATASET_DIR)
        self.recording = False
        self.current_label: Optional[str] = None
        self.current_data: List[np.ndarray] = []
        self._lock = threading.Lock()

    def start_recording(self, label: str):
        with self._lock:
            if self.recording:
                raise RuntimeError("Already recording")

            # Ensure label exists
            if label not in self.label_manager.get_labels():
                self.label_manager.add_label(label)

            self.current_label = label
            self.current_data = []
            self.recording = True
            print(f"Started recording for label: {label}")

    def stop_recording(self):
        with self._lock:
            if not self.recording:
                return

            self.recording = False
            label = self.current_label
            data = self.current_data
            self.current_label = None
            self.current_data = []

            if not data:
                print("No data collected")
                return

            self._save_data(label, data)
            print(f"Stopped recording. Saved {len(data)} frames.")

    def process_frame(self, frame_rgb, landmarks_list):
        """
        Called by the video stream loop when recording is active.
        landmarks_list: List[List[NormalizedLandmark]] from HandLandmarker
        """
        if not self.recording:
            return

        with self._lock:
            if landmarks_list:
                # Assuming single hand for now (first detected hand)
                res = landmarks_list[0]
                fv = hand_landmarks_to_feature_vector(res)
                label_idx = self.label_manager.get_label_index(self.current_label)
                self.current_data.append(append_label(fv, label_idx))

    def _save_data(self, label: str, data: List[np.ndarray]):
        created_time = int(time.time())
        config.DATASET_DIR.mkdir(parents=True, exist_ok=True)

        data_arr = np.asarray(data, dtype=np.float32)
        print(f"Saving raw data for {label}: {data_arr.shape}")
        np.save(config.DATASET_DIR / f"raw_{label}_{created_time}.npy", data_arr)

        # Create sequence data
        full_seq_data = []
        seq_length = config.SEQ_LENGTH
        if len(data_arr) >= seq_length:
            for start in range(0, len(data_arr) - seq_length + 1):
                full_seq_data.append(data_arr[start : start + seq_length])

            full_seq_arr = np.asarray(full_seq_data, dtype=np.float32)
            print(f"Saving seq data for {label}: {full_seq_arr.shape}")
            np.save(config.DATASET_DIR / f"seq_{label}_{created_time}.npy", full_seq_arr)
        else:
            print(f"Not enough frames for sequence (min {seq_length}), saved raw only.")

    def get_available_gestures(self) -> List[str]:
        return self.label_manager.get_labels()

    def add_gesture(self, label: str):
        self.label_manager.add_label(label)
