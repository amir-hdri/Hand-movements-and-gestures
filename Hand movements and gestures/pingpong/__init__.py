"""Bundled PingPong robot library (vendored).

Upstream code uses "flat" imports like `from connection.utils import Utils`.
To keep changes minimal while still allowing `import pingpong`, we ensure the
package directory is on `sys.path` so those legacy imports resolve.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

from .pingpongthread import PingPongThread  # noqa: E402

__all__ = ["PingPongThread"]

