# Hand Gesture Recognition System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-20232A?style=flat&logo=react)](https://reactjs.org/)
[![Material-UI](https://img.shields.io/badge/Material--UI-0081CB?style=flat&logo=mui)](https://mui.com/)

---

## 📋 Table of Contents

- [🚀 Features](#-features)
- [🏗️ Architecture](#️-architecture)
- [📦 Installation](#-installation)
- [🌐 Usage](#-usage)
- [🔧 Configuration](#-configuration)
- [📂 Project Structure](#-project-structure)
- [🤖 Running Tests](#-running-tests)
- [📜 API Documentation](#-api-documentation)
- [🛠️ Technologies](#-technologies)
- [💡 Contributing](#-contributing)
- [📄 License](#-license)

---

## 🚀 Features

### Real-Time Hand Gesture Recognition
- **Live Camera Feed**: Real-time hand tracking using MediaPipe
- **Multiple Gestures**: Support for custom gesture classification
- **High Accuracy**: Deep learning-based recognition with configurable thresholds

### Modern Web Interface
- **Dark/Light Mode**: Theme toggle with persistent preference
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Real-Time Visualization**: Live camera feed with landmarks overlay
- **Prediction History**: Track and review past predictions
- **Statistics Dashboard**: Visual analytics of gesture recognition performance

### Data Management
- **Dataset Collection**: Record and label hand movements
- **Model Training**: Train custom models with collected data
- **Export/Import**: Save and load datasets
- **Multi-Gesture Support**: Manage multiple gesture classes

### Advanced Features
- **Confidence Scoring**: Visual feedback on prediction accuracy
- **Stable Prediction**: Configurable stable count for reliable detection
- **Connection Monitoring**: Automatic reconnection handling
- **Error Recovery**: Graceful handling of camera and backend issues

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Client (React)                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐   │
│  │  CameraView │  │  Controls   │  │   Settings/Stats    │   │
│  └─────────────┘  └─────────────┘  └─────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Backend (FastAPI)                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐    │
│  │ Video Feed  │  │  REST API   │  │   Model Inference│    │
│  │  Endpoint   │  │  Endpoints  │  │   Engine         │    │
│  └─────────────┘  └─────────────┘  └─────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Processing Pipeline                       │
│  MediaPipe Hand Tracking → Feature Extraction → Model →      │
│  Gesture Classification → Prediction Output                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 Installation

### Prerequisites

- Python 3.11+
- Node.js 18+
- Git
- Web camera

### Clone the Repository

```bash
git clone https://github.com/amir-hdri/Hand-movements-and-gestures.git
cd Hand-movements-and-gestures/Hand movements and gestures
```

### Backend Setup

1. **Create a virtual environment** (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install Python dependencies**:

```bash
pip install -r requirements.txt
```

3. **Download MediaPipe model**:
   - The system will automatically download the `hand_landmarker.task` model on first run
   - Alternatively, manually download from [MediaPipe Model Garden](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)

> **Note on Keras 3 / legacy models:** the repo ships two models — `models/model.h5`
> (2 classes: `come`, `away`) and `models/model2_1.0.h5` (3 classes: `come`, `away`,
> `spin`). `model2_1.0.h5` was saved with Keras 2.4 and cannot be loaded directly by
> Keras 3. If you get an error like `Unrecognized keyword arguments passed to LSTM`,
> rebuild it once with:
>
> ```bash
> python scripts/convert_legacy_model.py
> ```

### Frontend Setup

1. **Navigate to frontend directory**:

```bash
cd gesture_recognition/gui/frontend
```

2. **Install Node.js dependencies**:

```bash
npm install
```

3. **Build the frontend**:

```bash
npm run build
```

The built frontend will be created in the `dist` directory.

---

## 🌐 Usage

### Run the Full System

```bash
# From the project root (Hand movements and gestures)
python run_gui.py
```

This will start the FastAPI backend on `http://localhost:8000` and serve the frontend.

- **Backend API**: `http://localhost:8000/api`
- **Frontend**: `http://localhost:8000/`

### Run Backend Only

```bash
# Navigate to backend directory
cd gesture_recognition/gui/backend

# Run with uvicorn
uvicorn app:app --reload
```

### Run Frontend Only (Development)

```bash
cd gesture_recognition/gui/frontend
npm run dev
```

This starts a development server on `http://localhost:5173` with hot reloading.

---

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the `gesture_recognition/gui/backend` directory:

```env
# CORS Settings (comma-separated origins)
CORS_ALLOWED_ORIGINS=http://localhost:5173,http://localhost:8000,http://127.0.0.1:5173,http://127.0.0.1:8000

# Dataset Directory
DATASET_DIR=./dataset

# Model Directory
MODELS_DIR=./models

# Sequence Length for Gesture Recognition
SEQ_LENGTH=30

# Confidence Threshold
CONFIDENCE_THRESHOLD=0.9

# Stable Count (consecutive predictions needed)
STABLE_COUNT=3
```

### Gesture Configuration

1. Start the system
2. Use the web interface to add new gestures
3. Record samples for each gesture
4. Train the model with your custom gestures

---

## 📂 Project Structure

```
Hand movements and gestures/
├── dataset/                      # Recorded gesture data
├── models/                       # Trained models
├── gesture_recognition/          # Core gesture recognition module
│   ├── __init__.py
│   ├── features.py              # Feature extraction from hand landmarks
│   ├── recognizer.py            # Gesture recognition logic
│   └── gui/                     # Web interface
│       ├── backend/             # FastAPI backend
│       │   ├── app.py           # Main API endpoints
│       │   ├── config.py        # Configuration
│       │   ├── data_manager.py  # Dataset management
│       │   └── inference.py     # Model inference
│       └── frontend/            # React frontend
│           ├── src/             # React components
│           │   ├── App.jsx       # Main application
│           │   ├── main.jsx      # Entry point
│           │   ├── theme.js      # Custom theme
│           │   ├── api.js        # API client
│           │   └── components/   # UI components
│           ├── package.json
│           └── vite.config.js
├── pingpong/                     # Robot control module (legacy)
│   ├── __init__.py
│   ├── connection/              # Connection handling
│   ├── operations/              # Robot operations
│   └── protocols/               # Communication protocols
├── tests/                        # Test suite
│   ├── test_*.py                # Various test files
│   └── conftest.py              # Pytest configuration
├── run_gui.py                   # Main entry point
├── robot.py                     # Robot control script
└── requirements.txt             # Python dependencies
```

---

## 🤖 Running Tests

### Run All Tests

```bash
# From the project root
python -m pytest tests/ -v
```

### Run Specific Tests

```bash
# Run only gesture recognition tests
python -m pytest tests/test_features.py tests/test_recognizer.py -v

# Run only pingpong tests
python -m pytest tests/pingpong/ -v

# Run with coverage
python -m pytest tests/ --cov=gesture_recognition --cov-report=html
```

### Test Configuration

Tests require the following dependencies:
- `pytest`
- `pytest-asyncio`
- `pytest-cov` (optional, for coverage)

Install test dependencies:

```bash
pip install pytest pytest-asyncio pytest-cov
```

---

## 📜 API Documentation

### Base URL

```
http://localhost:8000/api
```

### Endpoints

#### Camera & Status

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/video_feed` | MJPEG video stream from camera |
| GET | `/status` | Get current system status |

#### Gestures

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/gestures` | List all configured gestures |
| POST | `/gestures` | Add a new gesture |
| DELETE | `/gestures` | Delete a gesture |

#### Recording

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/record/start` | Start recording for a gesture |
| POST | `/record/stop` | Stop current recording |

#### Training

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/train` | Start model training |

#### Configuration

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/config` | Get current configuration (seq_length, threshold, stable_count, smart_thresholds) |
| POST | `/config` | Update configuration |

`GET /api/config` returns:

```json
{
  "seq_length": 30,
  "threshold": 0.9,
  "stable_count": 3,
  "smart_thresholds": { "stop": 0.98, "emergency": 0.99 }
}
```

`POST /api/config` accepts any subset of these fields and applies them
immediately (the recognizer is rebuilt so new values take effect):

```json
{ "seq_length": 40, "threshold": 0.85, "stable_count": 5 }
```

#### History & Dataset

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/history` | Get prediction history |
| DELETE | `/history` | Clear prediction history |
| POST | `/dataset/export` | Export dataset to ZIP |
| POST | `/dataset/reset` | Reset dataset (clear all data) |

### Request/Response Examples

**Start Recording:**
```bash
POST /api/record/start
Content-Type: application/json

{
  "label": "swipe_left"
}

Response:
{
  "status": "started",
  "label": "swipe_left"
}
```

**Get Status:**
```bash
GET /api/status

Response:
{
  "mode": "idle",
  "training_status": "idle",
  "last_prediction": {
    "action": "swipe_left",
    "confidence": 0.95
  }
}
```

---

## 🛠️ Technologies

### Backend
- **Python 3.11+** - Core language
- **FastAPI** - Modern web framework
- **Uvicorn** - ASGI server
- **MediaPipe** - Hand tracking and pose estimation
- **TensorFlow/Keras** - Deep learning framework
- **NumPy** - Numerical computing
- **OpenCV** - Computer vision
- **Pydantic** - Data validation

### Frontend
- **React 18** - UI framework
- **Material-UI (MUI)** - Component library
- **Framer Motion** - Animations
- **Recharts** - Data visualization
- **Notistack** - Snackbar notifications
- **Vite** - Build tool

### DevOps
- **pytest** - Testing framework
- **pytest-asyncio** - Async test support
- **GitHub Actions** - CI/CD (optional)

---

## 💡 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit your changes** (`git commit -m 'Add amazing feature'`)
4. **Push to the branch** (`git push origin feature/amazing-feature`)
5. **Open a Pull Request**

### Coding Standards

- Follow PEP 8 for Python code
- Use TypeScript for frontend code
- Write unit tests for new features
- Keep commits atomic and well-documented
- Update documentation as needed

### Pull Request Guidelines

- Provide a clear description of changes
- Include screenshots for UI changes
- Reference any related issues
- Ensure all tests pass
- Update README if documentation changes are needed

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Hand Gesture Recognition System

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🙏 Acknowledgments

- [MediaPipe](https://mediapipe.dev/) - Hand tracking technology
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
- [Material-UI](https://mui.com/) - UI component library
- [TensorFlow](https://www.tensorflow.org/) - Deep learning framework

---

## 📧 Contact

For questions, suggestions, or issues, please open a GitHub issue or contact the maintainers.

**Maintainer:** [amir-hdri](https://github.com/amir-hdri)

**GitHub Repository:** [https://github.com/amir-hdri/Hand-movements-and-gestures](https://github.com/amir-hdri/Hand-movements-and-gestures)

---

*Last updated: June 2026*
