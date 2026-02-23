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

    def load_data(self):
        data_list = []
        labels_list = []

        # Reload actions from config or labels.json if dynamic
        # Assuming DataManager updates config.ACTIONS or we read from labels.json
        # For now, let's just scan directory to find actions
        # But we need consistent label indices.
        # We should rely on the LabelManager logic.

        # Let's assume we pass the list of actions to train
        pass

    def train(self, actions: list[str], epochs=50):
        print(f"Training on actions: {actions}")

        # Collect data
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
                    # The saved seq data includes the label index as the last element of each frame?
                    # create_dataset.py:
                    # data.append(append_label(fv, label_index))
                    # then seq created from data.
                    # So shape is (N, 30, 100) where 100 = 99 features + 1 label
                    action_data.append(d)
                except Exception as e:
                    print(f"Error loading {f}: {e}")

            if action_data:
                full_action_data = np.concatenate(action_data, axis=0)
                # Extract features (exclude label from input)
                # The label in the file might be different if we re-indexed!
                # We should IGNORE the saved label index and use the current `idx`.

                # Input: all frames, all features except last
                x = full_action_data[:, :, :-1]
                # Create label array
                y = np.full((len(x),), idx)

                X_data.append(x)
                Y_data.append(y)

        if not X_data:
            raise ValueError("No training data found.")

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
