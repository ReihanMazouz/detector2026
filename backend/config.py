from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

APP_NAME = "Detector2026 API"
APP_VERSION = "0.1.0"
DEFAULT_FRONTEND_URL = "http://127.0.0.1:5174"
DEFAULT_BACKEND_PORT = 8001
DEFAULT_RUNS_ROOT = "./runs/"
RUNS_ROOT = PROJECT_ROOT / "runs"
