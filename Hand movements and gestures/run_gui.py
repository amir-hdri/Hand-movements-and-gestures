import uvicorn
import os
import sys
from pathlib import Path

# Add current directory to python path
sys.path.append(str(Path(__file__).resolve().parent))

if __name__ == "__main__":
    print("Starting Gesture Recognition GUI...")
    print("Backend API: http://localhost:8000/api")
    print("Frontend: http://localhost:8000/")

    # Check if frontend is built
    frontend_dist = Path("gesture_recognition/gui/frontend/dist")
    if not frontend_dist.exists():
        print("Warning: Frontend build not found!")
        print("To build frontend:")
        print("  cd gesture_recognition/gui/frontend")
        print("  npm install && npm run build")
        print("Then run this script again.")

    uvicorn.run("gesture_recognition.gui.backend.app:app", host="0.0.0.0", port=8000, reload=True)
