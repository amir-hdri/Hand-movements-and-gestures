import numpy as np
import os
import glob
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

class GestureTrainer:
    def __init__(self, dataset_dir="dataset", model_dir="models"):
        self.dataset_dir = Path(dataset_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.actions = []

    def load_data(self):
        """
        Load sequence data from .npy files.
        Infers actions from filenames: seq_{action}_{timestamp}.npy
        Returns: x_data, y_data, actions
        """
        seq_files = list(self.dataset_dir.glob("seq_*.npy"))
        if not seq_files:
            print("No dataset files found.")
            return None, None, []

        # Extract unique actions
        actions = set()
        for f in seq_files:
            # Filename format: seq_{action}_{timestamp}.npy
            parts = f.stem.split('_')
            # Handle action names with underscores? Assuming no underscores in action names for simplicity
            # parts[0] is 'seq', parts[-1] is timestamp. Middle is action.
            if len(parts) >= 3:
                action = "_".join(parts[1:-1])
                actions.add(action)

        self.actions = sorted(list(actions))
        print(f"Found actions: {self.actions}")

        x_data = []
        y_data = []

        for idx, action in enumerate(self.actions):
            action_files = self.dataset_dir.glob(f"seq_{action}_*.npy")
            for f in action_files:
                data = np.load(f)
                # data shape: (samples, seq_length, features)
                # Check if data already includes label (old format) or not
                # Old format: features + 1 (label)
                # New format (from my DataCollector): features only

                # Heuristic: Feature vector length is usually 99 or 100.
                # If shape[-1] is say 100, and we expect 99 features + 1 label.
                # Let's assume my DataCollector saves raw features.
                # But wait, original code has 100 features (99 features + 1 label).
                # landmarks_to_feature_vector returns 99? No, let's check features.py

                # For now, let's assume we use the data as is for X, and append label Y.
                # If the file was created by original script, it has the label at -1.
                # We should strip it if present.

                # Check feature dimension
                # If last dimension is 100, it likely includes label?
                # Actually, `hand_landmarks_to_feature_vector` returns a flat array.
                # Let's trust that we just want the features.

                # If we are training, we construct Y ourselves based on `idx`.

                # If the data has an extra column, remove it?
                # The original `train.ipynb` did: x_data = data[:, :, :-1] (took first 99 columns)
                # This implies 100 columns total.

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

        model = Sequential([
            LSTM(64, activation='relu', input_shape=x_train.shape[1:3]),
            Dense(32, activation='relu'),
            Dense(len(actions), activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

        checkpoint_path = self.model_dir / "model.h5"
        checkpoint = ModelCheckpoint(
            str(checkpoint_path),
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

        # Save actions list for inference
        import json
        with open(self.model_dir / "actions.json", "w") as f:
            json.dump(actions, f)

        return f"Training complete. Model saved to {checkpoint_path}. Actions: {actions}"
