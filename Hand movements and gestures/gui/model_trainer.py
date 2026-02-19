import numpy as np
import os
import glob
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from gesture_recognition.config import GestureConfig

class GestureTrainer:
    def __init__(self, config: GestureConfig):
        self.config = config
        self.dataset_dir = self.config.dataset_output_dir
        self.model_path = self.config.model_path
        # Model path is like 'models/model2_1.0.h5'. We need directory and filename separately if needed,
        # but Checkpoint can take the full path.
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.actions = []

    def load_data(self):
        """
        Load sequence data from .npy files.
        Infers actions from filenames: seq_{action}_{timestamp}.npy
        Returns: x_data, y_data, actions
        """
        seq_files = list(self.dataset_dir.glob("seq_*.npy"))
        if not seq_files:
            print(f"No dataset files found in {self.dataset_dir}.")
            return None, None, []

        # Extract unique actions
        actions = set()
        for f in seq_files:
            # Filename format: seq_{action}_{timestamp}.npy
            parts = f.stem.split('_')
            if len(parts) >= 3:
                action = "_".join(parts[1:-1])
                actions.add(action)

        self.actions = sorted(list(actions))
        print(f"Found actions in dataset: {self.actions}")

        # Verify against config actions? Optional, but good for debugging.
        # print(f"Configured actions: {self.config.actions}")

        x_data = []
        y_data = []

        for idx, action in enumerate(self.actions):
            action_files = self.dataset_dir.glob(f"seq_{action}_*.npy")
            for f in action_files:
                data = np.load(f)
                # data shape: (samples, seq_length, features)

                # Check for legacy label column
                if data.shape[-1] == 100:
                     x_data.append(data[:, :, :-1]) # Strip last column
                else:
                     x_data.append(data)

                # Create labels
                labels = np.full((data.shape[0],), idx, dtype=np.int32)
                y_data.append(labels)

        if not x_data:
            return None, None, []

        x_data = np.concatenate(x_data, axis=0)
        y_data = np.concatenate(y_data, axis=0)

        # One-hot encode
        y_data = to_categorical(y_data, num_classes=len(self.actions))

        return x_data.astype(np.float32), y_data.astype(np.float32), self.actions

    def train(self, epochs=50):
        x_data, y_data, actions = self.load_data()
        if x_data is None:
            return "No data found."

        print(f"Training on {x_data.shape[0]} samples. Actions: {actions}")

        x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.1, random_state=2021)

        input_shape = x_train.shape[1:3] # (seq_length, features)
        print(f"Input shape: {input_shape}")

        model = Sequential([
            LSTM(64, activation='relu', input_shape=input_shape),
            Dense(32, activation='relu'),
            Dense(len(actions), activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

        checkpoint = ModelCheckpoint(
            str(self.model_path),
            monitor='val_acc',
            verbose=1,
            save_best_only=True,
            mode='auto'
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_acc',
            factor=0.5,
            patience=10,
            verbose=1,
            mode='auto'
        )

        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            callbacks=[checkpoint, reduce_lr]
        )

        # Save actions list for inference alongside the model
        import json
        actions_path = self.model_path.parent / "actions.json"
        with open(actions_path, "w") as f:
            json.dump(actions, f)

        return f"Training complete. Model saved to {self.model_path}. Actions: {actions}"
