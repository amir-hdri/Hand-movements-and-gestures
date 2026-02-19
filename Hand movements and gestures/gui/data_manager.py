import cv2
import mediapipe as mp
import numpy as np
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from gesture_recognition.features import hand_landmarks_to_feature_vector
    from gesture_recognition.config import GestureConfig
except ImportError:
    print("Error: Could not import gesture_recognition package. Ensure proper directory structure.")
    # Simple mock for standalone testing if needed, though likely will fail later
    def hand_landmarks_to_feature_vector(landmarks):
        return np.zeros((99,), dtype=np.float32)
    GestureConfig = None

class DataCollector:
    def __init__(self, config: GestureConfig):
        self.config = config
        self.output_dir = self.config.dataset_output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.is_recording = False
        self.current_action = None
        self.recorded_data = []
        self.start_time = 0
        self.frame_count = 0

    def start_recording(self, action_name):
        self.is_recording = True
        self.current_action = action_name
        self.recorded_data = []
        self.start_time = time.time()
        self.frame_count = 0
        print(f"Started recording action: {action_name}")

    def stop_recording(self):
        if not self.is_recording:
            return 0

        self.is_recording = False
        count = len(self.recorded_data)
        if count > 0:
            self.save_data()
        print(f"Stopped recording. Captured {count} frames.")
        return count

    def process_frame(self, frame):
        """
        Process a single frame: detect hands, draw landmarks, collect data if recording.
        Returns: processed_frame, results
        """
        # Flip and convert
        img = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if results.multi_hand_landmarks:
            for res in results.multi_hand_landmarks:
                # Draw landmarks
                self.mp_drawing.draw_landmarks(img, res, self.mp_hands.HAND_CONNECTIONS)

                # Collect data
                if self.is_recording:
                    try:
                        fv = hand_landmarks_to_feature_vector(res)
                        self.recorded_data.append(fv)
                        self.frame_count += 1
                    except Exception as e:
                        print(f"Error extracting features: {e}")

        # Add overlay text
        if self.is_recording:
            cv2.putText(
                img,
                f"Recording: {self.current_action} ({self.frame_count})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        return img, results

    def save_data(self):
        """Save the recorded data to .npy files."""
        if not self.recorded_data:
            return

        timestamp = int(time.time())
        data_arr = np.array(self.recorded_data, dtype=np.float32)

        # Save raw data
        filename = self.output_dir / f"raw_{self.current_action}_{timestamp}.npy"
        np.save(filename, data_arr)
        print(f"Saved {filename}")

        # Create sequences using config.seq_length
        seq_length = self.config.seq_length
        if len(data_arr) >= seq_length:
            sequences = []
            for i in range(len(data_arr) - seq_length + 1):
                sequences.append(data_arr[i : i + seq_length])

            seq_arr = np.array(sequences, dtype=np.float32)
            seq_filename = self.output_dir / f"seq_{self.current_action}_{timestamp}.npy"
            np.save(seq_filename, seq_arr)
            print(f"Saved sequences to {seq_filename} shape={seq_arr.shape}")
        else:
            print(f"Not enough data for sequences (min {seq_length}).")
