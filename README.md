# Hand Movements and Gestures Control System

This project is a real-time gesture recognition system designed to control robots using hand movements captured via a webcam. It leverages machine learning to classify dynamic gestures and translate them into commands for hardware like PingPong robots, quadcopters, or robotic arms.

## Technologies Used

*   **Python**: The core programming language.
*   **OpenCV**: For real-time video capture and image processing.
*   **MediaPipe**: Google's framework for high-fidelity hand tracking and landmark detection.
*   **TensorFlow / Keras**: For building and training the LSTM (Long Short-Term Memory) neural network that classifies gesture sequences.
*   **Tkinter**: For the desktop Graphical User Interface (GUI).
*   **PySerial**: For communication with external hardware (PingPong robot).

## Features

1.  **Real-time Gesture Recognition**: Detects hand landmarks and classifies dynamic gestures (e.g., "come", "away", "spin") on the fly.
2.  **Interactive GUI**: A user-friendly desktop application to manage the entire workflow.
    *   **Inference & Control Tab**: View live recognition results and toggle robot control.
    *   **Data Collection Tab**: Easily record new gesture samples to expand the dataset.
    *   **Model Training Tab**: Retrain the AI model directly within the app after collecting new data.
    *   **Robot Control Tab**: Manual control panel for testing robot movements.
3.  **Robot Integration**:
    *   **PingPong Robot**: Direct support for the modular PingPong robot platform.
    *   **Extensible Interface**: Abstract `RobotController` class allows easy integration of other hardware (Quadcopters, Arms, etc.).
4.  **Mock Mode**: Simulates robot connection for development and testing without physical hardware.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd <repository-folder>
    ```

2.  **Install Dependencies**:
    The project requires Python 3.8+. Install the necessary packages:
    ```bash
    pip install -r "Hand movements and gestures/requirements.txt"
    ```
    *Note: If you are using Python 3.12+, ensure you use the updated requirements provided in this repo.*

## Usage

### Running the GUI Application

The easiest way to use the system is via the graphical interface:

```bash
python gui/app.py
```

### GUI Options & Workflow

1.  **Inference & Control**:
    *   **Video Feed**: Shows the webcam view with hand landmarks overlaid.
    *   **Prediction**: Displays the currently recognized gesture and confidence score.
    *   **Robot Mode**: Select "Mock" (simulated) or "PingPong" (real robot).
    *   **Connect Robot**: Establishes connection to the selected robot.

2.  **Data Collection**:
    *   **Action Name**: Enter a label for the new gesture (e.g., "takeoff", "land").
    *   **Start/Stop Recording**: Captures frames while you perform the gesture. Data is saved to `dataset/`.

3.  **Model Training**:
    *   **Train Model**: processing the collected data and training a new LSTM model.
    *   **Log**: Displays training progress and accuracy. The new model is automatically loaded for inference.

4.  **Robot Manual Control**:
    *   **Directional Pad**: Buttons to move the robot (Up, Down, Left, Right).
    *   **Actions**: Specific commands like Grab, Release, Takeoff, Land.

### Running CLI Scripts (Legacy)

You can still use the original command-line scripts:

*   **Data Collection**:
    ```bash
    python "Hand movements and gestures/create_dataset.py" --actions come away spin
    ```
*   **Real-time Recognition**:
    ```bash
    python "Hand movements and gestures/test.py"
    ```
*   **Robot Control (CLI)**:
    ```bash
    python "Hand movements and gestures/robot.py" --enable-robot
    ```

## Project Structure

*   `gui/`: Contains the desktop application code (`app.py`, `data_manager.py`, `model_trainer.py`, `robot_interface.py`).
*   `Hand movements and gestures/`: Core logic.
    *   `dataset/`: Stores recorded gesture data (`.npy` files).
    *   `models/`: Stores trained Keras models (`.h5`) and action labels (`.json`).
    *   `gesture_recognition/`: Feature extraction utilities.
    *   `pingpong/`: Driver code for PingPong robots.
