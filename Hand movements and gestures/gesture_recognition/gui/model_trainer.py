import numpy as np
import os
import glob
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, dataset_dir: Path = Path("dataset"), models_dir: Path = Path("models")):
        self.dataset_dir = dataset_dir
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.is_training = False

    def get_available_actions(self):
        # Scan dataset dir for seq_*.npy files
        files = glob.glob(str(self.dataset_dir / "seq_*.npy"))
        actions = set()
        for f in files:
            # format: seq_{action}_{timestamp}.npy
            filename = Path(f).name
            parts = filename.split("_")
            if len(parts) >= 3:
                action = "_".join(parts[1:-1])
                actions.add(action)
        return sorted(list(actions))

    def train_model(self, actions=None, epochs=50, model_name="new_model"):
        if self.is_training:
            return {"status": "error", "message": "Training already in progress"}

        self.is_training = True
        try:
            if actions is None:
                actions = self.get_available_actions()

            if not actions:
                return {"status": "error", "message": "No actions found in dataset"}

            logger.info(f"Training model for actions: {actions}")

            data_list = []
            for action in actions:
                action_files = glob.glob(str(self.dataset_dir / f"seq_{action}_*.npy"))
                if not action_files:
                    logger.warning(f"No data found for action: {action}")
                    continue

                for f in action_files:
                    data_list.append(np.load(f))

            if not data_list:
                return {"status": "error", "message": "No data loaded"}

            data = np.concatenate(data_list, axis=0)
            logger.info(f"Combined data shape: {data.shape}")

            x_data = data[:, :, :-1]
            labels = data[:, 0, -1]

            # Remap labels to 0..N-1 based on the 'actions' list order
            # The original labels in .npy might correspond to different indices if collected separately.
            # However, create_dataset.py stores the index provided at runtime.
            # If we train on a subset or different set, we must rely on the filename action name, not the stored label index.
            # Actually, `data[:, 0, -1]` contains the label index used during collection.
            # This is risky if 'come' was index 0 in one run and index 1 in another.
            # Better approach: Ignore stored label index. Construct y_data based on which file the data came from.

            # Let's redo data loading to ensure correct labels
            x_data_list = []
            y_data_list = []

            for idx, action in enumerate(actions):
                action_files = glob.glob(str(self.dataset_dir / f"seq_{action}_*.npy"))
                for f in action_files:
                    d = np.load(f)
                    x_data_list.append(d[:, :, :-1])
                    # Create labels for this batch
                    # d.shape[0] is number of samples
                    y = np.full((d.shape[0],), idx, dtype=int)
                    y_data_list.append(y)

            x_data = np.concatenate(x_data_list, axis=0)
            y_raw = np.concatenate(y_data_list, axis=0)

            y_data = to_categorical(y_raw, num_classes=len(actions))

            x_data = x_data.astype(np.float32)
            y_data = y_data.astype(np.float32)

            x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.1, random_state=2021)

            logger.info(f"Training shapes: {x_train.shape}, {y_train.shape}")

            model = Sequential([
                LSTM(64, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2])),
                Dense(32, activation='relu'),
                Dense(len(actions), activation='softmax')
            ])

            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

            model_path = str(self.models_dir / f"{model_name}.h5") # Keras 3 might prefer .keras, but .h5 is still supported usually

            # Using Keras 3, the format defaults to .keras if ending in .keras. .h5 uses legacy HDF5.
            # Let's stick to .keras for modern Keras, or .h5 if we want compatibility with older tools (though we are replacing them).
            # The prompt mentions "Legacy .h5 models ... are incompatible".
            # So let's use .keras for the new model to be safe with TF 2.16+.
            # Wait, if I use .keras, I need to make sure the Inference engine can load it.
            # load_model should handle it.

            if not model_name.endswith(".keras") and not model_name.endswith(".h5"):
                model_path = str(self.models_dir / f"{model_name}.keras")
            else:
                 model_path = str(self.models_dir / model_name)

            history = model.fit(
                x_train,
                y_train,
                validation_data=(x_val, y_val),
                epochs=epochs,
                callbacks=[
                    ModelCheckpoint(model_path, monitor='val_acc', verbose=1, save_best_only=True, mode='auto'),
                    ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=10, verbose=1, mode='auto')
                ]
            )

            # Save the action labels mapping alongside the model
            import json
            labels_path = str(self.models_dir / f"{model_name}_labels.json")
            with open(labels_path, "w") as f:
                json.dump(actions, f)

            return {
                "status": "success",
                "model_path": model_path,
                "accuracy": float(history.history['val_acc'][-1])
            }

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {"status": "error", "message": str(e)}
        finally:
            self.is_training = False
