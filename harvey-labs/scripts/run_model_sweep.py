#!/usr/bin/env python3
"""Compatibility wrapper for the model sweep CLI.

Prefer `uv run python utils/sweep.py` in new documentation.
"""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.sweep import main


if __name__ == "__main__":
    main()
