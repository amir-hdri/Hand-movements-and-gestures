import glob
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

from .config import config

class ModelTrainer:
    def __init__(self):
        self.dataset_dir = config.DATASET_DIR
        self.models_dir = config.MODELS_DIR
        self.actions = config.ACTIONS
        self.seq_length = config.SEQ_LENGTH

    def train(self, actions: list[str], epochs=50):
        print(f"Training on actions: {actions}")

        # Collect data.  We only include actions that actually have sequence
        # files, but we keep the index relative to the *full* actions list so
        # the model output layer matches the gesture ordering the app uses.
        X_data = []
        Y_data = []

        for idx, action in enumerate(actions):
            # Find all seq files for this action
            pattern = str(self.dataset_dir / f"seq_{action}_*.npy")
            files = glob.glob(pattern)

            if not files:
                print(f"Warning: No data found for action '{action}'")
                continue

            action_data = []
            for f in files:
                print(f"Loading {f}")
                try:
                    d = np.load(f)
                    action_data.append(d)
                except Exception as e:
                    print(f"Error loading {f}: {e}")

            if action_data:
                full_action_data = np.concatenate(action_data, axis=0)
                # Input: all frames, all features except the trailing label col
                x = full_action_data[:, :, :-1]
                # We IGNORE the saved label index and use the current `idx` so
                # the mapping always matches the gestures list in use.
                y = np.full((len(x),), idx)

                X_data.append(x)
                Y_data.append(y)

        if not X_data:
            raise ValueError("No training data found.")

        # At least two classes are required for a meaningful classifier.
        if len(X_data) < 2:
            raise ValueError(
                "At least 2 gestures with recorded data are required for training. "
                f"Only found data for: {[a for a in actions if any(self.dataset_dir.glob(f'seq_{a}_*.npy'))]}"
            )

        X = np.concatenate(X_data, axis=0).astype(np.float32)
        Y = np.concatenate(Y_data, axis=0).astype(int)

        # One-hot encode labels
        Y = to_categorical(Y, num_classes=len(actions))

        # Split
        x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.1, random_state=2021)

        # Build Model
        model = Sequential([
            LSTM(64, activation='relu', input_shape=x_train.shape[1:3]),
            Dense(32, activation='relu'),
            Dense(len(actions), activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

        self.models_dir.mkdir(parents=True, exist_ok=True)
        model_path = str(self.models_dir / config.MODEL_NAME)

        callbacks = [
            ModelCheckpoint(model_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max'),
            ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=10, verbose=1, mode='max')
        ]

        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            callbacks=callbacks
        )

        return history
