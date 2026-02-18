# Gesture Control GUI Application

This project now includes a Graphical User Interface (GUI) for easy data collection, model training, and robot control.

## Features

1.  **Inference & Control**: Real-time gesture recognition with visual feedback.
2.  **Data Collection**: Easily record new gestures by typing a name and clicking "Start Recording".
3.  **Model Training**: Retrain the model directly from the app after collecting new data.
4.  **Robot Control**: Manual control panel for Quadcopters/Robotic Arms and PingPong robots.

## Installation

Ensure you have installed the requirements:

```bash
pip install -r "Hand movements and gestures/requirements.txt"
```

## Running the App

Execute the application from the root directory:

```bash
python gui/app.py
```

## Usage Guide

### 1. Data Collection
- Go to the "Data Collection" tab.
- Enter a name for the gesture (e.g., "takeoff", "land").
- Click "Start Recording".
- Perform the gesture in front of the camera.
- Click "Stop Recording".
- Repeat for other gestures.

### 2. Training
- Go to the "Model Training" tab.
- Click "Train Model".
- Wait for the process to complete. The log will show accuracy.

### 3. Inference & Robot Control
- Go back to "Inference & Control".
- The system will now recognize your new gestures.
- Toggle "Robot Mode" to "PingPong" if you have the hardware, or stick to "Mock" for testing.
- Go to "Robot Manual Control" tab to send specific commands manually.

## Troubleshooting

- **Camera not found**: The app will fallback to a "Mock Camera" mode (green dot moving on black background).
- **Model loading error**: If the pre-trained model fails to load (due to version mismatch), simply train a new model using the "Model Training" tab.
