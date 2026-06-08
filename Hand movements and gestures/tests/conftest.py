import sys
import os
from pathlib import Path

# Add the project root to sys.path to fix import issues with spaces in directory name
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Also add the pingpong directory to sys.path for legacy flat imports
pingpong_dir = project_root / "pingpong"
if str(pingpong_dir) not in sys.path:
    sys.path.insert(0, str(pingpong_dir))

# Mock serial module before any imports that might need it
import unittest.mock
sys.modules['serial'] = unittest.mock.MagicMock()
sys.modules['serial.tools'] = unittest.mock.MagicMock()
sys.modules['serial.tools.list_ports'] = unittest.mock.MagicMock()
sys.modules['serial.threaded'] = unittest.mock.MagicMock()
