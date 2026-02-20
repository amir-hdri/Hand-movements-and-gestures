import cv2
import numpy as np
import time
import mediapipe as mp
from pathlib import Path
import logging
from gesture_recognition.features import append_label, hand_landmarks_to_feature_vector

logger = logging.getLogger(__name__)

class DatasetManager:
    def __init__(self, output_dir: Path = Path("dataset")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.recording = False
        self.current_action = None
        self.current_label_index = 0
        self.start_time = 0
        self.data_buffer = []
        self.seq_length = 30
        self.secs_for_action = 30

    def start_recording(self, action_name: str, label_index: int, duration: int = 30):
        self.current_action = action_name
        self.current_label_index = label_index
        self.secs_for_action = duration
        self.data_buffer = []
        self.start_time = time.time()
        self.recording = True
        logger.info(f"Started recording for action: {action_name} (index {label_index})")

    def stop_recording(self):
        if not self.recording:
            return

        self.recording = False
        self._save_data()
        self.current_action = None
        logger.info("Stopped recording")

    def process_frame(self, frame, hands, mp_drawing, mp_hands):
        """
        Process a frame for data collection.
        Returns the annotated frame.
        """
        if not self.recording:
            return frame

        elapsed = time.time() - self.start_time
        if elapsed > self.secs_for_action:
            self.stop_recording()
            return frame

        img = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for res in result.multi_hand_landmarks:
                fv = hand_landmarks_to_feature_vector(res)
                self.data_buffer.append(append_label(fv, self.current_label_index))
                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

        # Overlay recording status
        cv2.putText(
            img,
            f"Recording {self.current_action.upper()}: {int(self.secs_for_action - elapsed)}s",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        return img

    def _save_data(self):
        if not self.data_buffer:
            logger.warning("No data collected")
            return

        created_time = int(time.time())
        action = self.current_action

        data_arr = np.asarray(self.data_buffer, dtype=np.float32)
        logger.info(f"Saving raw data for {action}: {data_arr.shape}")
        np.save(self.output_dir / f"raw_{action}_{created_time}.npy", data_arr)

        # Create sequence data
        full_seq_data = []
        for start in range(0, len(data_arr) - self.seq_length + 1):
            full_seq_data.append(data_arr[start : start + self.seq_length])

        full_seq_arr = np.asarray(full_seq_data, dtype=np.float32)
        logger.info(f"Saving sequence data for {action}: {full_seq_arr.shape}")
        np.save(self.output_dir / f"seq_{action}_{created_time}.npy", full_seq_arr)
