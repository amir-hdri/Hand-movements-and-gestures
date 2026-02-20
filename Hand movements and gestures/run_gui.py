import sys
import os
import uvicorn
from pathlib import Path

def main():
    # Assuming this script is at the root of the "Hand movements and gestures" directory
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    print(f"Starting Gesture Recognition GUI...")
    print(f"Backend running on http://0.0.0.0:8000")

    # Import app after modifying sys.path
    try:
        from gesture_recognition.gui.app import app
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except ImportError as e:
        print(f"Error importing app: {e}")
        print("Please ensure dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)

if __name__ == "__main__":
    main()
