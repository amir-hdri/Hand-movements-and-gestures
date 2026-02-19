# Hand Gesture Recognition System

A comprehensive Hand Gesture Recognition system featuring a graphical user interface (GUI), real-time inference, model training, and robot control integration.

## Features

*   **Real-time Gesture Recognition:** Detects and classifies hand gestures using MediaPipe and LSTM models.
*   **Expanded Gesture Set:** Supports "come", "away", "spin", "thumbs_up", "peace", "fist", "stop".
*   **Graphical User Interface (GUI):**
    *   **Inference & Control:** View camera feed, recognition results, and control robot.
    *   **Data Collection:** Easily record new gesture samples.
    *   **Model Training:** Train custom models directly from the app.
    *   **Robot Control:** Manual and gesture-based control for PingPong robots (and mocks).
*   **Smart Thresholding:** Critical actions (like "stop") require higher confidence to prevent false positives.
*   **Robustness:** Improved error handling and configuration management.

## Project Structure

```text
gesture-recognition-system/
├── gesture_recognition/         # Core package (config, features, recognition pipeline)
├── gui/                         # GUI application (tkinter)
├── models/                      # Trained models (.h5) and actions metadata
├── dataset/                     # Collected gesture data (.npy)
├── pingpong/                    # PingPong robot communication library
├── robot.py                     # Headless robot control script
├── scripts/                     # Utility scripts (smoke tests, etc.)
├── tests/                       # Unit tests
├── requirements.txt             # Dependencies
└── README.md
```

## Setup

1.  **Create a Virtual Environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: This project supports Python 3.12. If you encounter issues with legacy models, you may need to retrain the model using the GUI.*

## Usage

### 1. Graphical User Interface (Recommended)

Launch the full-featured application:

```bash
python3 -m gui.app
```
*Make sure to run this from the project root (`Hand movements and gestures/`).*

**Workflow:**
*   **Inference:** The default tab shows the camera and recognition.
*   **Data Collection:** Go to the "Data Collection" tab. Enter an action name (e.g., "thumbs_up") and click "Start Recording". Perform the gesture in front of the camera.
*   **Training:** After collecting data for all actions, go to "Model Training" and click "Train Model". This will create a new model compatible with your current environment.
*   **Robot:** Connect a PingPong robot (if available) or use the Mock controller.

### 2. Headless Robot Control

For running on a robot without a display:

```bash
python3 robot.py --enable-robot
```

Options:
*   `--camera`: Specify camera index (default 0).
*   `--mock-robot`: Use a mock robot controller for testing.
*   `--model`: Path to a custom model file.

## Configuration

Core configuration is located in `gesture_recognition/config.py`. You can modify:
*   **Gestures:** Add or remove actions in `GestureConfig.actions`.
*   **Thresholds:** Adjust `threshold` and `critical_threshold`.
*   **Camera:** Change default `camera_id`.

## Troubleshooting

*   **Model Loading Error:** If you see errors about Keras version mismatch, use the "Model Training" tab in the GUI to train a fresh model on your machine.
*   **Camera Not Found:** Ensure your webcam is connected. The GUI will fall back to a "Mock" camera if none is found.
*   **PingPong Library:** Ensure the `pingpong` folder is in the python path (it is by default in this structure).

## Development

Run tests:
```bash
python3 -m unittest discover -s tests
```
