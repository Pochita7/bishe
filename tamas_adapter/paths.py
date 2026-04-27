"""Path helpers for TAMAS data and generated tool modules."""

from __future__ import annotations

import os
import sys


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAMAS_DATA_DIR = os.path.join(BASE_DIR, "TAMAS", "data")
TAMAS_TOOLS_DIR = os.path.join(TAMAS_DATA_DIR, "tools", "autogen")


def ensure_tamas_tools_path() -> str:
    """Expose generated TAMAS tool modules for importlib imports."""
    if os.path.isdir(TAMAS_TOOLS_DIR) and TAMAS_TOOLS_DIR not in sys.path:
        sys.path.insert(0, TAMAS_TOOLS_DIR)
    return TAMAS_TOOLS_DIR
