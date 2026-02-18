# Hand Gesture Recognition

Hand gesture recognition based on LSTM, MediaPipe Hands, and TensorFlow/Keras.

## Project Structure

```text
gesture-recognition-master/
├── create_dataset.py            # Webcam dataset collection
├── test.py                      # Realtime gesture recognition (webcam/video)
├── robot.py                     # Optional PingPong robot control
├── gesture_recognition/         # Core feature extraction + recognizer logic
├── pingpong/                    # Vendored PingPong robot library
├── dataset/                     # Sample dataset (.npy)
├── models/                      # Pretrained model (.h5)
├── notebooks/
│   └── train.ipynb              # Training notebook
├── scripts/
│   ├── check.sh                 # Compile + unit tests + smoke test
│   ├── smoke.py                 # Runtime/model load validation
│   └── generate_class_report_fa.py
├── tests/                       # Unit tests
├── artifacts/                   # Generated outputs (videos, reports, temp files)
├── requirements.txt
└── pyproject.toml
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python3 create_dataset.py --help
python3 test.py --help
python3 robot.py --help
```

Default realtime output videos are saved under `./artifacts/videos`.

## Validation

```bash
bash scripts/check.sh
```
