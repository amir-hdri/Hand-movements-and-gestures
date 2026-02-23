# Hand Gesture Recognition System

A comprehensive system for real-time hand gesture recognition using MediaPipe and TensorFlow. This repository includes a modern web-based application (React + FastAPI) as well as a legacy Tkinter desktop application.

## Features

- **Real-time Recognition**: Detects and classifies hand gestures using a webcam.
- **Smart Thresholding**: Configurable confidence thresholds for critical actions (e.g., "STOP").
- **Robot Control**: Integrated interface for controlling external hardware (PingPong robot, etc.).
- **Dataset Collection**: Tools to record and label new gesture data.
- **Model Training**: Built-in support for training custom LSTM models.

## Project Structure

```text
.
├── gesture_recognition/         # Core package
│   ├── gui/
│   │   ├── backend/             # FastAPI backend (Python)
│   │   └── frontend/            # React frontend (JS/JSX)
│   ├── features.py              # Feature extraction logic
│   └── recognizer.py            # Gesture classification logic
├── legacy_gui/                  # Legacy Tkinter desktop application
├── dataset/                     # Training data (.npy files)
├── models/                      # Trained models (.h5)
├── scripts/                     # Utility scripts (smoke tests, etc.)
├── run_gui.py                   # Launcher for the modern Web App
└── requirements.txt             # Python dependencies
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/amir-hdri/Hand-movements-and-gestures.git
    cd "Hand movements and gestures"
    ```

2.  **Set up Python environment (Python 3.12 recommended):**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Install Frontend Dependencies (Node.js required):**
    ```bash
    cd gesture_recognition/gui/frontend
    npm install
    npm run build
    cd ../../../..
    ```

## Running the Application

### 1. Modern Web Application (Recommended)

This launches the FastAPI backend which serves the React frontend.

```bash
python3 run_gui.py
```

*   **Access the App:** Open `http://localhost:8000` in your browser.
*   **API Docs:** Available at `http://localhost:8000/docs`.

### 2. Legacy Desktop Application

If you prefer the older Tkinter-based interface:

```bash
python3 legacy_gui/app.py
```

### 3. Command Line Tools

You can also run specific utilities directly:

*   **Real-time Inference (CLI):**
    ```bash
    python3 test.py --source 0 --model models/model.h5
    ```

*   **Smoke Test (Verify Setup):**
    ```bash
    python3 scripts/smoke.py
    ```

## Dataset & Training

The `dataset/` directory contains pre-recorded gesture data (`.npy` files).

*   **To collect new data:** Use the "Data Collection" tab in the Web App or Legacy App.
*   **To train a model:** Use the "Training" tab in the Web App. The new model will be saved to `models/`.

## Configuration

*   **Backend Config:** `gesture_recognition/gui/backend/config.py` allows you to modify gesture names, thresholds, and sequence lengths.
*   **Default Gestures:** `come`, `away`.

## Troubleshooting

*   **"Model output mismatch":** Ensure `config.ACTIONS` matches the number of classes your model was trained on. The default `model.h5` expects 2 classes.
*   **"Keras 3.x detected":** The system supports Keras 3 but includes logic to handle legacy Keras 2 models. If issues persist, try reinstalling dependencies.
